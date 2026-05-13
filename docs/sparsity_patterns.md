# Sparsity Patterns in XSAKE

XSAKE supports five block-sparse attention patterns, all expressed as a boolean `block_mask[H, n_q_blocks, n_kv_blocks]`. Each pattern is a different prior on which (query_block, key_block) pairs are relevant.

## Pattern Taxonomy

### Dense

```
block_mask[h, i, j] = True  for all j ≤ i   (causal)
```

Used as the correctness baseline. Zero sparsity, full quadratic complexity.

### Local / Sliding Window

```
block_mask[h, i, j] = True  iff  i - window_blocks ≤ j ≤ i
```

Each query block only attends to the `window_blocks` most recent KV blocks. Efficient for tasks with short-range dependencies (e.g., POS tagging). Configured via `hads_config.yaml:local_window_blocks`.

### BigBird

Combines three components:
- **Global tokens**: first and last KV block always active
- **Local window**: 2-block sliding window
- **Random blocks**: ~10% of remaining KV blocks per query

This approximates full attention while keeping complexity O(n√n) in expectation.

### Random

Uniform random sparsity: each query block attends to exactly `ceil((1 - sparsity_ratio) × n_kv_blocks)` randomly chosen KV blocks, always including the causal diagonal.

Used as a null hypothesis in ablations to separate structured vs random sparsity gains.

### HADS (Head-Adaptive Dynamic Sparsity)

See `docs/hads_design.md`. Per-head entropy-driven mask built during training calibration.

## Block Size Trade-offs

| block_size | Memory alignment | Sparsity granularity | Overhead |
|------------|-----------------|---------------------|----------|
| 32 | Poor on A100 | Fine | High loop count |
| 64 | Good | Medium | Balanced |
| **128** | **Optimal on T4/A100** | **Coarse** | **Low** |
| 256 | Excellent | Very coarse | Very low |

Default: `block_size=128`. Configurable in `configs/kernel_config.yaml`.

## Correctness Invariants

All patterns must satisfy:
1. `block_mask[h, i, j] == False` for all `j > i` (causal)
2. `block_mask[h, i, i] == True` for all `i` (self-attention)
3. Shape: `bool[n_heads, n_q_blocks, n_kv_blocks]`

These are verified in `tests/test_kernel_correctness.py`.

## Adding a New Pattern

Implement in `kernels/pallas/sparsity_mask.py`:

```python
def make_block_mask(seq_len, block_size, sparsity_type="my_pattern", **kwargs):
    # ... existing patterns ...
    elif sparsity_type == "my_pattern":
        n = seq_len // block_size
        mask = np.zeros((n, n), dtype=bool)
        # fill mask — must be causal (upper triangle = False)
        return jnp.array(mask)
```

Then add `"my_pattern"` to the benchmark config and CI regression thresholds.
