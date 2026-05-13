# Abandoned: Naive pmap Sharding

## What was attempted

Launched training with naive `jax.pmap` — full model + optimizer state
replicated on every GPU, batch split across devices.

## What broke

`jax.errors.ResourceExhaustedError: Out of memory on device`

On T4 (16GB) with batch=16, seq=2048, 12 layers, d_model=768:

| Component                     | Memory     |
|-------------------------------|------------|
| Model params (fp32)           | ~350 MB    |
| Adam moments (fp32)           | ~700 MB    |
| Activations (12 layers, bf16) | ~4.8 GB    |
| Attention score matrix        | ~1.6 GB    |
| Gradients                     | ~350 MB    |
| **Total**                     | **~7.8 GB** |

XLA overhead pushed this past 16GB → OOM.

## Root causes

1. Attention score matrix is O(seq²) — `[16,12,2048,2048]` × 4 bytes = 1.6 GB
2. No gradient checkpointing: all 12 layers stored simultaneously for backward
3. fp32 Adam: two moment tensors = 2× param memory

## Fix

Three changes together:
1. `nn.remat` gradient checkpointing → O(1) activation memory, 30% slower backward
2. HADS sparsity → score matrix shrinks ~45%
3. Per-device batch 16 → 8
