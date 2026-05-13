"""
End-to-end kernel latency benchmark across batch sizes and sequence lengths.

Sweeps (seq_len × batch_size) and reports median latency with confidence intervals.
Generates benchmarks/results/latency_curves.png.

Run on Kaggle T4:
    python -m benchmarks.latency_benchmark
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kernels.pallas.sparse_attention import sparse_attention_reference
from kernels.hads.hads_pattern import dummy_hads_profile


def _latency_ms(fn, *args, n_warmup=5, n_trials=30) -> tuple[float, float]:
    """Return (median_ms, std_ms)."""
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*args))

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.median(times)), float(np.std(times))


def run_latency_benchmark(
    seq_lens: List[int] = (512, 1024, 2048, 4096),
    batch_sizes: List[int] = (1, 2, 4),
    n_heads: int = 12,
    d_head: int = 64,
    block_size: int = 128,
) -> dict:
    backend = jax.default_backend()
    results = {}
    print(f"\n[benchmark] Latency sweep — backend: {backend}")

    for seq in seq_lens:
        results[seq] = {}
        for bs in batch_sizes:
            key = jax.random.PRNGKey(0)
            q = jax.random.normal(key, (bs, n_heads, seq, d_head), jnp.bfloat16)
            k = jax.random.normal(key, (bs, n_heads, seq, d_head), jnp.bfloat16)
            v = jax.random.normal(key, (bs, n_heads, seq, d_head), jnp.bfloat16)

            @jax.jit
            def dense_fn(q, k, v):
                s = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (d_head ** -0.5)
                w = jax.nn.softmax(s, axis=-1)
                return jnp.einsum("bhqk,bhkd->bhqd", w, v)

            profile = dummy_hads_profile(n_heads=n_heads, seq_len=seq, block_size=block_size)
            bm = profile.block_masks

            @jax.jit
            def xsake_fn(q, k, v):
                out, _ = sparse_attention_reference(q, k, v, bm, block_size)
                return out

            dense_med, dense_std   = _latency_ms(dense_fn, q, k, v)
            xsake_med, xsake_std   = _latency_ms(xsake_fn, q, k, v)
            speedup = dense_med / xsake_med if xsake_med > 0 else 1.0

            results[seq][bs] = {
                "dense_ms":  round(dense_med, 3),
                "dense_std": round(dense_std, 3),
                "xsake_ms":  round(xsake_med, 3),
                "xsake_std": round(xsake_std, 3),
                "speedup":   round(speedup, 3),
            }
            print(f"  seq={seq:4d} bs={bs} | dense={dense_med:.2f}ms "
                  f"| xsake={xsake_med:.2f}ms | {speedup:.2f}×")

    return {"backend": backend, "results": results}


def plot_latency_curves(results: dict, out_path: str) -> None:
    seq_lens = sorted(int(k) for k in results["results"])
    batch_sizes = sorted(set(
        int(bs) for r in results["results"].values() for bs in r
    ))

    fig, axes = plt.subplots(1, len(batch_sizes), figsize=(5 * len(batch_sizes), 5), sharey=False)
    if len(batch_sizes) == 1:
        axes = [axes]

    colors = {"dense": "#4C72B0", "xsake": "#DD8452"}

    for ax, bs in zip(axes, batch_sizes):
        dense_times  = [results["results"][str(s)][str(bs)]["dense_ms"] for s in seq_lens]
        xsake_times  = [results["results"][str(s)][str(bs)]["xsake_ms"] for s in seq_lens]
        dense_stds   = [results["results"][str(s)][str(bs)]["dense_std"] for s in seq_lens]
        xsake_stds   = [results["results"][str(s)][str(bs)]["xsake_std"] for s in seq_lens]

        ax.errorbar(seq_lens, dense_times, yerr=dense_stds, label="Dense JAX",
                    color=colors["dense"], marker="o", linewidth=2, capsize=4)
        ax.errorbar(seq_lens, xsake_times, yerr=xsake_stds, label="XSAKE (HADS)",
                    color=colors["xsake"], marker="s", linewidth=2, capsize=4)

        ax.set_title(f"batch_size={bs}", fontsize=12)
        ax.set_xlabel("Sequence length")
        ax.set_ylabel("Latency (ms)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("XSAKE vs Dense Attention Latency", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_lens",   nargs="+", type=int, default=[512, 1024, 2048, 4096])
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--out_json",   default="benchmarks/results/latency_results.json")
    parser.add_argument("--out_plot",   default="benchmarks/results/latency_curves.png")
    args = parser.parse_args()

    results = run_latency_benchmark(args.seq_lens, args.batch_sizes)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    plot_latency_curves(results, args.out_plot)
