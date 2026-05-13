# XSAKE Kernel Design

## Overview

XSAKE implements a block-sparse attention kernel stack on JAX + Pallas. The design targets three goals: (1) correctness on CPU for local testing, (2) maximum throughput on GPU via Pallas→Triton compilation, (3) zero user-visible branching between backends.

## Kernel Layers

```
sparse_attention()          ← unified entry point
  ├─ sparse_attention_pallas()   ← Pallas kernel (GPU/TPU)
  │    └─ _sparse_attn_fwd_kernel()  ← fori_loop over KV blocks
  └─ sparse_attention_reference()    ← pure JAX (CPU fallback)
```

### Block Decomposition

Sequences are divided into blocks of `block_size` tokens. The attention computation becomes a loop over `n_q_blocks × n_kv_blocks` pairs, each controlled by a boolean `block_mask[h, q_block, kv_block]`.

```
n_blocks = seq_len // block_size           # e.g. 1024 // 128 = 8
block_mask: bool[n_heads, n_blocks, n_blocks]
```

Skipped blocks (mask=False) save both FLOPs and HBM bandwidth because neither the KV tile load nor the matmul is issued.

### Online Softmax (FlashAttention-style)

The Pallas kernel accumulates output using running max/sum statistics, eliminating the need to materialize the full `S × S` attention matrix:

```python
# per KV-block iteration inside fori_loop:
scores_block = q_tile @ k_block.T / sqrt(D)
m_new = max(m_prev, rowmax(scores_block))
exp_block = exp(scores_block - m_new)
sum_new = sum_prev * exp(m_prev - m_new) + rowsum(exp_block)
out_acc  = out_acc  * exp(m_prev - m_new) + exp_block @ v_block
```

This is numerically stable and composes with the block mask: when `block_mask[h, i, j] == False`, the iteration is still executed (to keep the loop trip count static for XLA) but writes zeros into `exp_block`.

**Note**: A future optimization (tracked in F-002 / `failure_analysis/FAILURES.md`) would use dynamic trip count via `lax.while_loop`, but this breaks XLA's static shape requirements and is deferred.

## Fused Kernels

| Kernel | File | Notes |
|--------|------|-------|
| `fused_softmax` | `kernels/pallas/fused_softmax.py` | Max+exp+normalize in a single Pallas pass |
| `fused_layernorm` | `kernels/pallas/fused_layernorm.py` | Mean+var+normalize+scale+shift fused |

Both kernels use `pl.load` / `pl.store` with explicit SRAM tiling and fall back to JAX primitives on CPU.

## XLA Compiler Flags

`kernels/xla/compiler_flags.py` sets backend-specific flags at process start:

- **GPU**: `--xla_gpu_enable_triton_softmax_fusion`, `--xla_gpu_autotune_level=4`, cuBLAS workspace
- **TPU**: `--xla_tpu_enable_flash_attention`, `--xla_tpu_enable_sparse_core_matmul`
- **CPU**: minimal flags, disables spurious warnings

These are set once via `os.environ["XLA_FLAGS"]` before `jax.config` is imported.

## Kernel Versioning and HLO Inspection

`kernels/xla/fusion_pass.py` provides `inspect_hlo(fn, *args)` which runs the function, captures the optimized HLO (high-level operations) text, and checks for expected fusion patterns. This is used in CI to verify that layernorm and softmax appear as `kCustomFusionOp` entries rather than separate kernel launches.

## Performance Targets (GPU T4, seq=4096)

| Metric | Dense Baseline | XSAKE (HADS) | Target |
|--------|---------------|--------------|--------|
| Attention latency | ~185 ms | ~110 ms | 35–45% reduction |
| HBM usage | 100% | ~45% | 40–60% reduction |
| MFU | ~28% | ~41% | — |
