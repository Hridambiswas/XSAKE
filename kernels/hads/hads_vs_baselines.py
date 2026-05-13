"""Compare HADS against BigBird, Longformer, Sliding Window, and Random sparsity."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import List

import jax
import jax.numpy as jnp
import numpy as np

from kernels.hads.hads_pattern import dummy_hads_profile
from kernels.pallas.sparse_attention import sparse_attention_reference
from kernels.pallas.sparsity_mask import make_block_mask


@dataclass
class BaselineResult:
    method: str
    seq_len: int
    active_block_fraction: float
    median_latency_ms: float
    latency_reduction_pct: float
    kl_vs_dense: float
    memory_reduction_pct: float


def _time_ms(fn, *args, n_warmup: int = 3, n_trials: int = 10) -> float:
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def _kl(p: jnp.ndarray, q: jnp.ndarray, eps: float = 1e-8) -> float:
    p = jnp.clip(p, eps, None)
    q = jnp.clip(q, eps, None)
    return float(jnp.mean(jnp.sum(p * jnp.log(p / q), axis=-1)))


def _build_method_masks(
    method: str,
    n_heads: int,
    seq_len: int,
    block_size: int,
    hads_profile=None,
) -> jnp.ndarray:
    n_blocks = seq_len // block_size
    if method == "dense":
        masks = np.ones((n_heads, n_blocks, n_blocks), dtype=bool)
    elif method == "hads":
        masks = np.stack([hads_profile.block_masks[h] for h in range(n_heads)])
    elif method == "sliding_window":
        window_blocks = 2
        mask = np.zeros((n_blocks, n_blocks), dtype=bool)
        for i in range(n_blocks):
            for j in range(max(0, i - window_blocks), i + 1):
                mask[i, j] = True
        masks = np.broadcast_to(mask, (n_heads, n_blocks, n_blocks)).copy()
    elif method == "bigbird":
        # Global tokens (first/last block) + sliding window (2 blocks) + random (10%)
        mask = np.zeros((n_blocks, n_blocks), dtype=bool)
        rng = np.random.default_rng(0)
        for i in range(n_blocks):
            mask[i, 0] = True
            mask[i, -1] = True
            for j in range(max(0, i - 1), i + 1):
                mask[i, j] = True
            rand_cols = rng.choice(n_blocks, size=max(1, n_blocks // 10), replace=False)
            mask[i, rand_cols] = True
            mask[i, i + 1 :] = False  # causal
        masks = np.broadcast_to(mask, (n_heads, n_blocks, n_blocks)).copy()
    elif method == "random":
        rng = np.random.default_rng(0)
        masks = np.zeros((n_heads, n_blocks, n_blocks), dtype=bool)
        for h in range(n_heads):
            for i in range(n_blocks):
                masks[h, i, i] = True  # always attend to self
                active = rng.choice(i + 1, size=max(1, (i + 1) // 2), replace=False)
                masks[h, i, active] = True
    elif method == "longformer":
        window_blocks = 1
        mask = np.zeros((n_blocks, n_blocks), dtype=bool)
        for i in range(n_blocks):
            for j in range(max(0, i - window_blocks), i + 1):
                mask[i, j] = True
        mask[:, 0] = True  # global token: first block
        masks = np.broadcast_to(mask, (n_heads, n_blocks, n_blocks)).copy()
    else:
        raise ValueError(f"Unknown method: {method}")
    return jnp.array(masks)


def run_comparison(
    seq_lens: List[int] | None = None,
    n_heads: int = 12,
    d_head: int = 64,
    block_size: int = 128,
    batch_size: int = 1,
) -> List[BaselineResult]:
    if seq_lens is None:
        seq_lens = [512, 1024, 2048]

    methods = ["dense", "hads", "sliding_window", "bigbird", "longformer", "random"]
    results = []

    for seq_len in seq_lens:
        print(f"\n--- seq_len={seq_len} ---")
        key = jax.random.PRNGKey(0)
        B, H, S, D = batch_size, n_heads, seq_len, d_head
        q = jax.random.normal(key, (B, H, S, D), jnp.bfloat16)
        k = jax.random.normal(key, (B, H, S, D), jnp.bfloat16)
        v = jax.random.normal(key, (B, H, S, D), jnp.bfloat16)

        hads_profile = dummy_hads_profile(n_heads=H, seq_len=S, block_size=block_size)

        # Get dense baseline timing
        dense_masks = _build_method_masks("dense", H, S, block_size)
        dense_fn = jax.jit(lambda q, k, v: sparse_attention_reference(q, k, v, dense_masks, block_size)[0])
        dense_ms = _time_ms(dense_fn, q, k, v)
        dense_out = dense_fn(q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32))
        dense_p = jax.nn.softmax(dense_out, axis=-1)

        for method in methods:
            masks = _build_method_masks(method, H, S, block_size, hads_profile)
            fn = jax.jit(lambda q, k, v, m=masks: sparse_attention_reference(q, k, v, m, block_size)[0])
            ms = _time_ms(fn, q, k, v)
            out = fn(q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32))
            kl = _kl(dense_p, jax.nn.softmax(out, axis=-1))
            active_frac = float(jnp.mean(masks))
            mem_reduction = (1.0 - active_frac) * 100.0
            latency_reduction = (dense_ms - ms) / dense_ms * 100.0
            r = BaselineResult(
                method=method,
                seq_len=seq_len,
                active_block_fraction=active_frac,
                median_latency_ms=ms,
                latency_reduction_pct=latency_reduction,
                kl_vs_dense=kl,
                memory_reduction_pct=mem_reduction,
            )
            results.append(r)
            print(f"  {method:15s}  active={active_frac:.2f}  lat={ms:.2f}ms  gain={latency_reduction:.1f}%  KL={kl:.4f}")

    return results


def save_comparison(results: List[BaselineResult], path: str) -> None:
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nSaved {len(results)} results to {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_lens", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--out", type=str, default="benchmarks/results/hads_vs_baselines.json")
    args = parser.parse_args()

    print("=== HADS vs Baselines Comparison ===")
    results = run_comparison(seq_lens=args.seq_lens)
    save_comparison(results, args.out)
