# XSAKE — XLA-Fused Sparse Attention Kernel Engine

A production-grade distributed training infrastructure built on JAX + Pallas, featuring a novel **Head-Adaptive Dynamic Sparsity (HADS)** algorithm for efficient long-sequence transformers.

---

## Key Results

| Metric | Dense Baseline | XSAKE (HADS) | Target |
|--------|---------------|--------------|--------|
| Attention latency (seq=4096) | 713.8 ms | **412.4 ms** | 35–45% ↓ |
| HBM memory usage (seq=4096) | 1207.6 MB | **484.4 MB** | 40–60% ↓ |
| KL divergence vs dense | 0.000 | **0.089** | < 0.5 |
| Perplexity vs dense | baseline | **+1.8%** | < 5% |
| Training loss (20 steps) | — | 5.54 → 4.78 | converging |

> Hardware: NVIDIA T4 (Kaggle). `[projected]` entries in full benchmark report use linear scaling with 95% efficiency.

---

## What is HADS?

Standard sparse attention patterns (BigBird, Longformer) assign the **same mask to every head**. But attention heads specialize:

- **Semantic heads** focus on a few key tokens (low entropy) → can skip most KV blocks
- **Syntactic heads** attend broadly (high entropy) → need nearly full attention

HADS measures per-head Shannon entropy from a calibration pass and assigns per-head sparsity ratios via a monotone-decreasing linear map:

```
ρ_h = max_sparsity − (H_h − H_min)/(H_max − H_min) × (max_sparsity − min_sparsity)
```

Result: focused heads skip 80–90% of blocks; broad heads skip only 10–20%.

```
Head entropy → sparsity ratio (12 heads, seq=1024):

Head 0  ████████████░░░░░░░  (62% skip)
Head 1  ██████░░░░░░░░░░░░░  (82% skip)  ← syntactic
Head 2  ████████████████░░░  (18% skip)  ← semantic
Head 3  █████████░░░░░░░░░░  (71% skip)
...
```

---

## Architecture

```
XSAKE/
├── kernels/
│   ├── pallas/          ← Block-sparse attention (JAX Pallas → Triton/XLA)
│   │   ├── sparse_attention.py    # Core kernel with online softmax
│   │   ├── fused_softmax.py       # Fused max+exp+normalize
│   │   └── fused_layernorm.py     # Fused mean+var+scale+shift
│   ├── hads/            ← Head-Adaptive Dynamic Sparsity (novel contribution)
│   │   ├── hads_pattern.py        # Entropy measurement + mask construction
│   │   ├── hads_ablation.py       # Sensitivity studies
│   │   └── hads_vs_baselines.py   # Comparison vs BigBird/Longformer/etc
│   ├── triton/          ← FlashAttention-2 baseline
│   └── xla/             ← Compiler flags + HLO fusion inspection
├── model/               ← GPT-style transformer (Flax/linen)
│   ├── attention.py     # XSAKEAttention with HADS integration
│   ├── transformer.py   # Full model with weight tying + remat
│   └── config.py        # ModelConfig, HADSConfig, TrainingConfig dataclasses
├── distributed/         ← Multi-device training harness
│   ├── mesh.py          # 2D device mesh (data × model parallel)
│   ├── sharding.py      # PartitionSpecs + shard utilities
│   └── pmap_trainer.py  # pmap step with pmean gradient sync
├── training/
│   ├── trainer.py       # Main train loop + HADS recalibration
│   ├── optimizer.py     # AdamW + cosine warmup + weight decay masking
│   ├── checkpointing.py # Orbax versioned checkpoints
│   └── mixed_precision.py # bf16 forward, fp32 master weights
├── data/                ← OpenWebText pipeline (HuggingFace + GPT-2 BPE)
├── benchmarks/          ← Latency, memory, throughput, HADS ablation
├── observability/       ← FastAPI dashboard + Prometheus + W&B
├── registry/            ← Experiment + model artifact tracking
├── ci/                  ← Regression detection thresholds + reporter
├── tests/               ← pytest suite (kernel correctness, HADS, sharding)
├── docs/                ← Design docs (kernel, HADS, sparsity, distributed)
├── failure_analysis/    ← 6 documented failures + lessons learned
└── scripts/             ← Training, benchmarking, profiling shell scripts
```

---

## Sparsity Pattern Comparison (seq=2048)

| Method | Active blocks | Latency | KL vs dense |
|--------|--------------|---------|-------------|
| Dense | 100% | 181.2 ms | 0.000 |
| Sliding Window | 28% | 64.3 ms | 0.312 |
| Longformer | 32% | 71.8 ms | 0.278 |
| BigBird | 41% | 84.2 ms | 0.198 |
| Random 50% | 50% | 103.7 ms | 0.421 |
| **HADS** | **43%** | **105.6 ms** | **0.089** |

HADS achieves the best output quality (lowest KL) among all sparse methods by adapting to per-head entropy.

---

## Quick Start

### Local (CPU / Mac M4)

```bash
git clone https://github.com/Hridambiswas/XSAKE && cd XSAKE
pip install -e ".[dev]"

# Run kernel correctness tests
pytest tests/test_kernel_correctness.py tests/test_hads_pattern.py -v

# Smoke training (tiny model, 20 steps)
pytest tests/test_loss_convergence.py -v

# HADS ablation (CPU-compatible)
bash scripts/run_hads_ablation.sh
```

### GPU (Kaggle T4)

```bash
pip install -e ".[training,observability]"
BACKEND=gpu bash scripts/run_benchmarks.sh
BACKEND=gpu STEPS=10000 bash scripts/run_training.sh
```

### Observability Dashboard

```bash
pip install -e ".[observability]"
uvicorn observability.dashboard.app:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

### Docker

```bash
# GPU
docker build -f docker/Dockerfile.gpu -t xsake:gpu .
docker run --gpus all -p 8000:8000 xsake:gpu

# TPU
docker build -f docker/Dockerfile.tpu -t xsake:tpu .
```

---

## HADS Theory (Quick Reference)

Full derivation in `kernels/hads/hads_theory.md`.

```
1. Calibration pass (50 batches):
   A[b,h,i,j] = softmax(QK^T/√D)[b,h,i,j]

2. Entropy per head:
   H_h = -1/S Σ_i Σ_j A[h,i,j] log(A[h,i,j] + ε)

3. Sparsity ratio (monotone decreasing in entropy):
   ρ_h = ρ_max − (H_h − H_min)/(H_max − H_min) × (ρ_max − ρ_min)
   where ρ_min = 0.10, ρ_max = 0.90

4. Block mask:
   - Causal constraint: mask[h,i,j] = False for j > i  (always)
   - Global token:      mask[h,i,0] = True              (always)
   - Skip fraction ρ_h of remaining KV blocks per head

5. Recalibrate every 500 training steps.
```

---

## CI / CD

Three GitHub Actions workflows:

| Workflow | Trigger | What it checks |
|----------|---------|---------------|
| `kernel_correctness.yml` | Every PR | Kernel output vs reference, shape/dtype, causal correctness |
| `training_smoke_test.yml` | Push to main | 20-step convergence, loss < 12.0 and decreasing |
| `benchmark_regression.yml` | Kernel/model changes | HADS latency/memory/KL vs thresholds; PR comment on failure |

---

## Failure Analysis

Six documented failure modes in `failure_analysis/FAILURES.md`:

| ID | Failure | Resolution |
|----|---------|-----------|
| F-001 | Static sparsity hurt perplexity 8% | → Switched to HADS entropy-adaptive |
| F-002 | Triton compilation failures on Mac | → Switched to Pallas (Triton-backend agnostic) |
| F-003 | pmap OOM with large batch | → Added `shard_batch_pmap` shape validation |
| F-004 | BlockSpec dimension mismatch | → Fixed grid_spec alignment to block_size |
| F-005 | Causal mask not enforced in sparse path | → Added upper-triangle check in `make_block_mask` |
| F-006 | bf16 underflow in gradient accumulation | → Added `check_finite` + fp32 master weights |

---

## Showcase

**Live**: https://xsake-hridam.vercel.app

| Page | Description |
|------|-------------|
| [Home](https://xsake-hridam.vercel.app) | Hero, key results, HADS explanation, baseline comparison table |
| [HADS Visualizer](https://xsake-hridam.vercel.app/hads) | Per-head block mask grid, entropy/sparsity sliders, head classification |
| [Benchmarks](https://xsake-hridam.vercel.app/benchmarks) | Latency / memory / throughput / KL charts with raw data tables |
| [Training Replay](https://xsake-hridam.vercel.app/training) | Animated 10k-step loss curve, HADS recalibration event timeline |

Built with Next.js 16 (App Router, fully static), Recharts, Tailwind CSS.

---

## Citation

```bibtex
@misc{biswas2026xsake,
  title  = {XSAKE: XLA-Fused Sparse Attention with Head-Adaptive Dynamic Sparsity},
  author = {Biswas, Hridam},
  year   = {2026},
  url    = {https://github.com/Hridambiswas/XSAKE}
}
```

---

## License

MIT
