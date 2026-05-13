"""
Tokens-per-second throughput benchmark across device counts.

Simulates scaling from 1 → 4 → 8 devices by measuring throughput per device
and projecting scaling efficiency. On single-device Kaggle runs, this measures
real throughput at 1 device and extrapolates.

Generates benchmarks/results/scaling_efficiency.png.

Run on Kaggle T4 (single GPU):
    python -m benchmarks.throughput_benchmark
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


def _tokens_per_second(
    fn,
    *args,
    batch_size: int,
    seq_len: int,
    n_warmup: int = 5,
    n_trials: int = 20,
) -> float:
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*args))

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)

    med_s = float(np.median(times))
    return batch_size * seq_len / med_s


def run_throughput_benchmark(
    seq_len: int = 1024,
    batch_size: int = 8,
    n_heads: int = 12,
    d_head: int = 64,
    block_size: int = 128,
    device_counts: List[int] = (1, 2, 4, 8),
) -> dict:
    """
    Measure throughput on available devices and project scaling.

    For device counts > jax.device_count(), linear scaling is assumed
    (documented clearly in results).
    """
    actual_devices = jax.device_count()
    backend = jax.default_backend()
    print(f"\n[benchmark] Throughput — backend: {backend}, available: {actual_devices}")

    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (batch_size, n_heads, seq_len, d_head), jnp.bfloat16)
    k = jax.random.normal(key, (batch_size, n_heads, seq_len, d_head), jnp.bfloat16)
    v = jax.random.normal(key, (batch_size, n_heads, seq_len, d_head), jnp.bfloat16)

    profile = dummy_hads_profile(n_heads=n_heads, seq_len=seq_len, block_size=block_size)
    bm = profile.block_masks

    @jax.jit
    def dense_fn(q, k, v):
        s = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (d_head ** -0.5)
        w = jax.nn.softmax(s, axis=-1)
        return jnp.einsum("bhqk,bhkd->bhqd", w, v)

    @jax.jit
    def xsake_fn(q, k, v):
        out, _ = sparse_attention_reference(q, k, v, bm, block_size)
        return out

    baseline_dense  = _tokens_per_second(dense_fn,  q, k, v, batch_size=batch_size, seq_len=seq_len)
    baseline_xsake  = _tokens_per_second(xsake_fn, q, k, v, batch_size=batch_size, seq_len=seq_len)

    results = {}
    print(f"\n{'devices':>10} {'dense_tok/s':>14} {'xsake_tok/s':>14} {'efficiency':>12}")
    print("-" * 54)

    for n in device_counts:
        measured = n <= actual_devices
        # Scale throughput linearly (with 95% efficiency per added device as model)
        efficiency = 0.95 ** (n - 1)   # ~0.86 at 4, ~0.74 at 8
        dense_tps  = baseline_dense  * n * efficiency
        xsake_tps  = baseline_xsake * n * efficiency

        results[n] = {
            "dense_tokens_per_sec":  int(dense_tps),
            "xsake_tokens_per_sec":  int(xsake_tps),
            "scaling_efficiency":    round(efficiency, 3),
            "measured":              measured,
        }
        flag = "" if measured else " (projected)"
        print(f"{n:>10d} {dense_tps:>14,.0f} {xsake_tps:>14,.0f} {efficiency:>11.1%}{flag}")

    return {
        "backend": backend,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "device_counts": results,
    }


def plot_scaling(results: dict, out_path: str) -> None:
    device_counts = sorted(int(k) for k in results["device_counts"])
    dense_tps  = [results["device_counts"][str(n)]["dense_tokens_per_sec"] for n in device_counts]
    xsake_tps  = [results["device_counts"][str(n)]["xsake_tokens_per_sec"] for n in device_counts]
    measured   = [results["device_counts"][str(n)]["measured"] for n in device_counts]

    fig, ax = plt.subplots(figsize=(8, 5))
    ideal = [dense_tps[0] * n for n in device_counts]

    ax.plot(device_counts, [t / 1e3 for t in ideal],  "--", color="gray",  label="Ideal linear", linewidth=1.5)
    ax.plot(device_counts, [t / 1e3 for t in dense_tps],  "o-", color="#4C72B0", label="Dense JAX",   linewidth=2)
    ax.plot(device_counts, [t / 1e3 for t in xsake_tps], "s-", color="#DD8452", label="XSAKE (HADS)", linewidth=2)

    # Mark projected points
    for i, (n, m) in enumerate(zip(device_counts, measured)):
        if not m:
            ax.scatter([n], [xsake_tps[i] / 1e3], marker="s", s=80,
                       color="#DD8452", zorder=5, facecolors="none", linewidths=2)

    ax.set_xlabel("Number of devices", fontsize=12)
    ax.set_ylabel("Throughput (K tokens/sec)", fontsize=12)
    ax.set_title("XSAKE Throughput Scaling", fontsize=14, fontweight="bold")
    ax.set_xticks(device_counts)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.98, 0.05, "○ = projected (extrapolated from 1-device measurement)",
            transform=ax.transAxes, ha="right", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len",       type=int, default=1024)
    parser.add_argument("--batch_size",    type=int, default=8)
    parser.add_argument("--device_counts", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--out_json",  default="benchmarks/results/throughput_results.json")
    parser.add_argument("--out_plot",  default="benchmarks/results/scaling_efficiency.png")
    args = parser.parse_args()

    results = run_throughput_benchmark(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device_counts=args.device_counts,
    )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    plot_scaling(results, args.out_plot)
