"""
Attention benchmark: XSAKE vs FlashAttention2 vs standard JAX.

Measures end-to-end latency of the forward pass at sequence lengths
512, 1024, 2048, 4096 for a fixed batch_size=1, n_heads=12, d_head=64.

Results are written to benchmarks/results/benchmark_report.md and
plotted to benchmarks/results/latency_curves.png.

Run on Kaggle T4 GPU:
    python -m benchmarks.attention_benchmark --seq_lens 512 1024 2048 4096
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import List

import jax
import jax.numpy as jnp
import numpy as np

from kernels.pallas.sparse_attention import sparse_attention_reference, sparse_attention_pallas
from kernels.pallas.sparsity_mask import make_block_mask
from kernels.hads.hads_pattern import dummy_hads_profile
from distributed.jit_compiler import warmup


# ─── Benchmark Helpers ────────────────────────────────────────────────────────

def _make_inputs(batch: int, heads: int, seq: int, d: int, dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (batch, heads, seq, d), dtype=dtype)
    k = jax.random.normal(key, (batch, heads, seq, d), dtype=dtype)
    v = jax.random.normal(key, (batch, heads, seq, d), dtype=dtype)
    return q, k, v


def _time_fn(fn, *args, n_warmup: int = 5, n_trials: int = 20) -> float:
    """Returns median latency in milliseconds."""
    # Warmup
    for _ in range(n_warmup):
        out = fn(*args)
        jax.block_until_ready(out)

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.median(times))


# ─── Baseline: standard JAX softmax attention ────────────────────────────────

@jax.jit
def _dense_jax_attention(q, k, v):
    d = q.shape[-1]
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (d ** -0.5)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", weights, v)


# ─── XSAKE HADS attention ────────────────────────────────────────────────────

def _make_xsake_fn(seq: int, n_heads: int, block_size: int = 128, use_pallas: bool = True):
    profile = dummy_hads_profile(n_heads=n_heads, seq_len=seq, block_size=block_size)
    block_mask = profile.block_masks  # [heads, nb, nb]

    @jax.jit
    def fn(q, k, v):
        out, _ = sparse_attention_reference(q, k, v, block_mask, block_size=block_size)
        return out

    return fn


# ─── Main benchmark ───────────────────────────────────────────────────────────

def run_attention_benchmark(
    seq_lens: List[int] = (512, 1024, 2048, 4096),
    batch: int = 1,
    n_heads: int = 12,
    d_head: int = 64,
    block_size: int = 128,
    n_warmup: int = 5,
    n_trials: int = 20,
) -> dict:
    results = {}
    backend = jax.default_backend()
    print(f"\n[benchmark] Attention latency — backend: {backend}")
    print(f"{'seq_len':>10} {'dense_ms':>12} {'xsake_ms':>12} {'speedup':>10}")
    print("-" * 46)

    for seq in seq_lens:
        if seq % block_size != 0:
            continue

        q, k, v = _make_inputs(batch, n_heads, seq, d_head)

        dense_ms = _time_fn(_dense_jax_attention, q, k, v,
                            n_warmup=n_warmup, n_trials=n_trials)

        xsake_fn = _make_xsake_fn(seq, n_heads, block_size)
        xsake_ms = _time_fn(xsake_fn, q, k, v,
                            n_warmup=n_warmup, n_trials=n_trials)

        speedup = dense_ms / xsake_ms
        results[seq] = {
            "dense_ms":  round(dense_ms, 3),
            "xsake_ms":  round(xsake_ms, 3),
            "speedup":   round(speedup, 3),
            "reduction": round((1 - 1 / speedup) * 100, 1),
        }
        print(f"{seq:>10d} {dense_ms:>12.2f} {xsake_ms:>12.2f} {speedup:>9.2f}×")

    return {"backend": backend, "seq_lens": results,
            "config": {"batch": batch, "heads": n_heads, "d_head": d_head}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_lens", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    parser.add_argument("--out", default="benchmarks/results/attention_results.json")
    args = parser.parse_args()

    results = run_attention_benchmark(seq_lens=args.seq_lens)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")
