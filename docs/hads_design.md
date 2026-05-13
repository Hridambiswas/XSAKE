# HADS: Head-Adaptive Dynamic Sparsity

## Motivation

Fixed sparsity patterns (BigBird, Longformer, sliding window) assign the same block mask to every attention head. This ignores a well-known empirical phenomenon: attention heads specialize. Semantic heads (low entropy, attending to a few highly relevant tokens) tolerate high sparsity. Syntactic heads (high entropy, attending broadly) require dense access.

HADS measures per-head attention entropy from a calibration pass and uses it to assign per-head sparsity ratios dynamically.

## Algorithm

### Step 1: Entropy Measurement (Calibration)

Given calibration attention matrices `A ∈ ℝ^{B × H × S × S}`, compute head entropy:

```
H_h = -1/S Σ_{i=1}^{S} Σ_{j=1}^{S} A_{h,i,j} log(A_{h,i,j} + ε)
```

This is Shannon entropy averaged over query positions. High entropy ↔ diffuse attention ↔ syntactic/broad head. Low entropy ↔ focused attention ↔ semantic head.

### Step 2: Entropy → Sparsity Mapping

A monotone-decreasing linear map:

```
ρ_h = max_sparsity - (H_h - H_min) / (H_max - H_min) × (max_sparsity - min_sparsity)
```

- Lowest-entropy head → `max_sparsity` (most skipped blocks)
- Highest-entropy head → `min_sparsity` (fewest skipped blocks)

Default bounds: `min_sparsity=0.10`, `max_sparsity=0.90`.

### Step 3: Block Mask Construction

For each head `h` with sparsity ratio `ρ_h`, we build `block_mask[h] ∈ bool^{n_blocks × n_blocks}`:

1. Always set causal diagonal blocks active (`mask[i, j] = True` for `j ≤ i`)
2. Always set the first block column active (global token analog)
3. Randomly mask `floor(ρ_h × n_off_diag_blocks)` upper-triangle blocks to False

The resulting profile is stored as a `HADSProfile(NamedTuple)` containing `head_entropy`, `head_sparsity`, and `block_masks`.

## Properties

| Property | Value |
|----------|-------|
| Causal invariance | Always preserved (never masks i→j for j > i) |
| Sparsity bounds | [0.10, 0.90] per head |
| Monotone mapping | Entropy ↑ → sparsity ↓ (proven in `hads_theory.md`) |
| Recalibration frequency | Every 500 training steps |

## Comparison with Baselines

| Method | Per-head sparsity | Adaptive | Global tokens | Local window |
|--------|------------------|----------|---------------|--------------|
| Dense | 0% | No | Yes | Yes |
| Sliding Window | ~75% | No | No | Yes |
| Longformer | ~70% | No | Yes | Yes |
| BigBird | ~60% | No | Yes | Yes |
| **HADS** | 10–90% per head | **Yes** | Yes (first block) | Causal |

## Implementation Files

- `kernels/hads/hads_pattern.py` — core algorithm
- `kernels/hads/hads_theory.md` — mathematical derivation
- `kernels/hads/hads_ablation.py` — sensitivity studies
- `kernels/hads/hads_vs_baselines.py` — empirical comparison
- `model/attention.py` — integration with `XSAKEAttention`
- `training/trainer.py` — `_calibrate_hads()` recalibration loop

## Limitations

1. Entropy estimation requires a calibration forward pass (~50 batches). Cold-start uses `dummy_hads_profile()` with linearly-spaced synthetic entropy.
2. Block masks are static between recalibrations. Token-level dynamic routing (MoE-style) is a future direction.
3. On very short sequences (S < 256) the entropy signal is noisy; `min_sparsity` is clamped to 0.3 in this regime.
