# Abandoned: Triton Kernel First Attempt

## What was attempted

Initial implementation targeted a Triton-based sparse attention kernel
to match FlashAttention2's codegen approach.

## What broke

```
ModuleNotFoundError: No module named 'triton'
```

Triton requires CUDA + NVIDIA GPU. Dev machine is Mac M4 (Apple Silicon) — no CUDA.

## Why Pallas was chosen instead

1. **Same GPU codegen**: Pallas compiles to Triton on CUDA GPUs. Identical performance, no Triton source needed.
2. **TPU native**: Pallas runs on TPU via XLA. Triton requires a full rewrite.
3. **CPU fallback**: Pallas falls back to reference JAX on CPU — enables local correctness tests without GPU.
4. **JAX integration**: participates in jit/grad/vmap/pmap automatically. Triton requires custom_vjp wrapper.

## What was preserved

`kernels/triton/flash_baseline.py` keeps the FlashAttention2 Triton baseline
for GPU benchmark comparison on Kaggle T4.
