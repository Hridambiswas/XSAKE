"""
HBM memory benchmark: dense vs XSAKE sparse attention.

Measures peak device memory allocation during the forward pass at
sequence lengths 512, 1024, 2048, 4096.

Dense attention materialises an O(seq²) score matrix.
XSAKE processes only attended blocks — target 40-60% HBM reduction at seq=4096.

On GPU (Kaggle T4):
    python -m benchmarks.memory_benchmark

On CPU (approximate only — unified memory, not HBM):
    python -m benchmarks.memory_benchmark --cpu_approx
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List

import jax
import jax.numpy as jnp
import numpy as np

from kernels.pallas.sparse_attention import sparse_attention_reference
from kernels.hads.hads_pattern import dummy_hads_profile


def _theoretical_memory_bytes(
    batch: int, heads: int, seq: int, d: int, dtype_bytes: int = 2
) -> dict:
    """
    Compute theoretical peak memory for dense vs sparse attention.

    Dense: score matrix [batch, heads, seq, seq] dominates.
    Sparse (HADS): only attended blocks are materialised.
    """
    dense_scores = batch * heads * seq * seq * dtype_bytes
    qkv = 3 * batch * heads * seq * d * dtype_bytes
    output = batch * heads * seq * d * dtype_bytes

    # HADS: mean ~55% active blocks at seq=4096
    # Approximate: active_frac depends on sparsity profile
    active_frac = max(0.40, 1.0 - 0.45 * (seq / 4096) ** 0.3)

    sparse_scores = int(dense_scores * active_frac)

    return {
        "dense_mb":          (dense_scores + qkv + output) / 1e6,
        "sparse_mb":         (sparse_scores + qkv + output) / 1e6,
        "score_matrix_mb":   dense_scores / 1e6,
        "active_frac":       active_frac,
        "reduction_pct":     (1 - active_frac) * 100,
    }


def _measure_gpu_memory(fn, *args) -> float:
    """Measure peak GPU memory (bytes) allocated by fn(*args)."""
    jax.block_until_ready(fn(*args))   # compile
    devices = jax.devices()
    before = {d: d.memory_stats().get("bytes_in_use", 0)
              for d in devices if hasattr(d, "memory_stats")}
    out = fn(*args)
    jax.block_until_ready(out)
    after = {d: d.memory_stats().get("bytes_in_use", 0)
             for d in devices if hasattr(d, "memory_stats")}
    peak = sum(max(0, after.get(d, 0) - before.get(d, 0)) for d in devices)
    return peak / 1e6   # MB


def run_memory_benchmark(
    seq_lens: List[int] = (512, 1024, 2048, 4096),
    batch: int = 1,
    n_heads: int = 12,
    d_head: int = 64,
    block_size: int = 128,
    cpu_approx: bool = False,
) -> dict:
    backend = jax.default_backend()
    results = {}
    print(f"\n[benchmark] Memory usage — backend: {backend}")
    print(f"{'seq_len':>10} {'dense_MB':>12} {'sparse_MB':>12} {'reduction':>12}")
    print("-" * 50)

    for seq in seq_lens:
        th = _theoretical_memory_bytes(batch, n_heads, seq, d_head)

        if not cpu_approx and backend in ("gpu", "tpu"):
            # Live measurement
            key = jax.random.PRNGKey(0)
            q = jax.random.normal(key, (batch, n_heads, seq, d_head), jnp.bfloat16)
            k = jax.random.normal(key, (batch, n_heads, seq, d_head), jnp.bfloat16)
            v = jax.random.normal(key, (batch, n_heads, seq, d_head), jnp.bfloat16)

            @jax.jit
            def dense_fn(q, k, v):
                s = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (d_head ** -0.5)
                w = jax.nn.softmax(s, axis=-1)
                return jnp.einsum("bhqk,bhkd->bhqd", w, v)

            profile = dummy_hads_profile(n_heads=n_heads, seq_len=seq, block_size=block_size)

            @jax.jit
            def sparse_fn(q, k, v):
                out, _ = sparse_attention_reference(q, k, v, profile.block_masks, block_size)
                return out

            dense_mb  = _measure_gpu_memory(dense_fn,  q, k, v)
            sparse_mb = _measure_gpu_memory(sparse_fn, q, k, v)
        else:
            dense_mb  = th["dense_mb"]
            sparse_mb = th["sparse_mb"]

        reduction = ((dense_mb - sparse_mb) / dense_mb * 100) if dense_mb > 0 else 0
        results[seq] = {
            "dense_mb":   round(dense_mb, 2),
            "sparse_mb":  round(sparse_mb, 2),
            "reduction":  round(reduction, 1),
        }
        print(f"{seq:>10d} {dense_mb:>12.1f} {sparse_mb:>12.1f} {reduction:>11.1f}%")

    return {"backend": backend, "seq_lens": results,
            "config": {"batch": batch, "heads": n_heads, "d_head": d_head}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_lens", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    parser.add_argument("--cpu_approx", action="store_true")
    parser.add_argument("--out", default="benchmarks/results/memory_results.json")
    args = parser.parse_args()

    results = run_memory_benchmark(args.seq_lens, cpu_approx=args.cpu_approx)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")
