# Benchmark Methodology

## Principles

1. **Median over mean**: All latency results report the median of N trials. Medians are robust to JIT compilation spikes and GC pauses.
2. **Warmup mandatory**: Every benchmark runs ≥3 warmup iterations (discarded) before timing. This ensures HBM caches are warm and JIT compilation has fired.
3. **`jax.block_until_ready()`**: All timings wrap the call in `jax.block_until_ready()` to account for JAX's async dispatch.
4. **Seed fixed**: All random inputs use `jax.random.PRNGKey(0)` for reproducibility.
5. **No overhead from Python-side work**: Input arrays are pre-created outside the timed loop.

## Benchmark Suite

### `benchmarks/attention_benchmark.py`
Measures median attention latency (ms) for dense JAX vs XSAKE sparse at sequence lengths 512, 1024, 2048, 4096. Reports latency reduction %.

### `benchmarks/memory_benchmark.py`
Reports theoretical and live HBM usage. Theoretical: `4 × n_heads × seq^2 × 4 bytes` for fp32 attention matrix. Live: `device_memory_stats()` delta before/after kernel execution.

### `benchmarks/latency_benchmark.py`
2D sweep over `seq_lens × batch_sizes`. Generates latency curves with error bars (±1 std). Output: `latency_curves.png`.

### `benchmarks/throughput_benchmark.py`
Tokens/sec = `(batch_size × seq_len) / latency_s`. Reports scaling efficiency for multi-device: `efficiency = throughput_N / (N × throughput_1)`. Projected linearly for devices beyond available hardware.

### `benchmarks/hads_benchmark.py`
Compares HADS vs 4 baselines (dense, sliding window, BigBird, random) across:
- Active block fraction (sparsity proxy)
- Latency (ms)
- KL divergence vs dense output (quality proxy)
- Estimated HBM savings

## CI Regression Thresholds (`ci/regression_thresholds.yaml`)

| Metric | Threshold | Action |
|--------|-----------|--------|
| Latency regression | >5% slower | Fail CI |
| Memory regression | >3% more memory | Fail CI |
| Active block reduction | <35% vs dense | Fail CI |
| KL divergence | >0.5 vs dense | Fail CI |

Regressions post a comment on the PR via `ci/regression_reporter.py` and exit with code 1.

## Projected vs Measured Results

Results in `benchmarks/results/` are labeled:
- `measured`: run on actual hardware (Kaggle T4)
- `projected`: extrapolated from single-device measurements using linear scaling with 95% efficiency assumption

The README table clearly marks which entries are projected.

## Reproducing Benchmarks

```bash
# Full suite (requires GPU)
BACKEND=gpu bash scripts/run_benchmarks.sh

# HADS ablation only (CPU-compatible)
bash scripts/run_hads_ablation.sh

# Kernel profiler + TensorBoard trace
PROFILE_DIR=/tmp/xsake_profile bash scripts/profile_kernel.sh
tensorboard --logdir /tmp/xsake_profile
```
