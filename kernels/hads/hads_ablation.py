"""HADS ablation study: sweep entropy thresholds and sparsity bounds."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from kernels.hads.hads_pattern import (
    HADSProfile,
    build_hads_profile,
    dummy_hads_profile,
    entropy_to_sparsity,
)
from kernels.pallas.sparse_attention import sparse_attention_reference


@dataclass
class AblationConfig:
    seq_len: int = 1024
    block_size: int = 128
    n_heads: int = 12
    d_head: int = 64
    batch_size: int = 1
    n_warmup: int = 3
    n_trials: int = 10
    entropy_scales: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    min_sparsity_values: Tuple[float, ...] = (0.1, 0.2, 0.3)
    max_sparsity_values: Tuple[float, ...] = (0.7, 0.8, 0.9)


@dataclass
class AblationResult:
    entropy_scale: float
    min_sparsity: float
    max_sparsity: float
    mean_sparsity: float
    active_block_fraction: float
    median_latency_ms: float
    latency_vs_dense_pct: float
    kl_vs_dense: float


def _time_fn(fn, *args, n_warmup: int = 3, n_trials: int = 10) -> float:
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def _kl_divergence(p: jnp.ndarray, q: jnp.ndarray, eps: float = 1e-8) -> float:
    p = jnp.clip(p, eps, None)
    q = jnp.clip(q, eps, None)
    return float(jnp.mean(jnp.sum(p * jnp.log(p / q), axis=-1)))


def _run_single_ablation(
    key: jax.Array,
    cfg: AblationConfig,
    entropy_scale: float,
    min_sparsity: float,
    max_sparsity: float,
) -> AblationResult:
    B, H, S, D = cfg.batch_size, cfg.n_heads, cfg.seq_len, cfg.d_head
    q = jax.random.normal(key, (B, H, S, D), jnp.bfloat16)
    k = jax.random.normal(key, (B, H, S, D), jnp.bfloat16)
    v = jax.random.normal(key, (B, H, S, D), jnp.bfloat16)

    # Build synthetic profile with scaled entropy
    base_profile = dummy_hads_profile(n_heads=H, seq_len=S, block_size=cfg.block_size)
    scaled_entropy = base_profile.head_entropy * entropy_scale
    sparsity_ratios = entropy_to_sparsity(
        scaled_entropy,
        min_sparsity=min_sparsity,
        max_sparsity=max_sparsity,
    )

    # Rebuild block masks with new sparsity ratios
    n_blocks = S // cfg.block_size
    rng = np.random.default_rng(42)
    block_masks = []
    for h in range(H):
        n_skip = int(sparsity_ratios[h] * n_blocks)
        mask = np.ones((n_blocks, n_blocks), dtype=bool)
        # Always keep causal diagonal blocks
        skip_candidates = [(i, j) for i in range(n_blocks) for j in range(i + 1, n_blocks)]
        rng.shuffle(skip_candidates)
        for i, j in skip_candidates[:n_skip]:
            mask[i, j] = False
        block_masks.append(jnp.array(mask))
    block_masks_arr = jnp.stack(block_masks)

    # Dense reference
    dense_masks = jnp.ones((H, n_blocks, n_blocks), dtype=bool)
    dense_fn = jax.jit(lambda q, k, v: sparse_attention_reference(q, k, v, dense_masks, cfg.block_size)[0])
    sparse_fn = jax.jit(lambda q, k, v: sparse_attention_reference(q, k, v, block_masks_arr, cfg.block_size)[0])

    dense_ms = _time_fn(dense_fn, q, k, v, n_warmup=cfg.n_warmup, n_trials=cfg.n_trials)
    sparse_ms = _time_fn(sparse_fn, q, k, v, n_warmup=cfg.n_warmup, n_trials=cfg.n_trials)

    # Correctness proxy: KL divergence on output logits (fp32 for stability)
    dense_out = dense_fn(q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32))
    sparse_out = sparse_fn(q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32))
    p = jax.nn.softmax(dense_out, axis=-1)
    q_dist = jax.nn.softmax(sparse_out, axis=-1)
    kl = _kl_divergence(p, q_dist)

    active_fraction = float(jnp.mean(block_masks_arr))
    mean_sparsity = float(jnp.mean(sparsity_ratios))

    return AblationResult(
        entropy_scale=entropy_scale,
        min_sparsity=min_sparsity,
        max_sparsity=max_sparsity,
        mean_sparsity=mean_sparsity,
        active_block_fraction=active_fraction,
        median_latency_ms=sparse_ms,
        latency_vs_dense_pct=100.0 * (dense_ms - sparse_ms) / dense_ms,
        kl_vs_dense=kl,
    )


def run_ablation(cfg: AblationConfig | None = None) -> List[AblationResult]:
    if cfg is None:
        cfg = AblationConfig()
    key = jax.random.PRNGKey(0)
    results = []
    for es in cfg.entropy_scales:
        for min_s in cfg.min_sparsity_values:
            for max_s in cfg.max_sparsity_values:
                if min_s >= max_s:
                    continue
                print(f"  entropy_scale={es:.1f}  min_s={min_s:.2f}  max_s={max_s:.2f}", end="  ", flush=True)
                r = _run_single_ablation(key, cfg, es, min_s, max_s)
                print(f"latency_gain={r.latency_vs_dense_pct:.1f}%  KL={r.kl_vs_dense:.4f}")
                results.append(r)
    return results


def save_results(results: List[AblationResult], path: str) -> None:
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Saved {len(results)} results to {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--out", type=str, default="benchmarks/results/hads_ablation.json")
    args = parser.parse_args()

    cfg = AblationConfig(seq_len=args.seq_len)
    print(f"=== HADS Ablation Study (seq_len={cfg.seq_len}) ===")
    results = run_ablation(cfg)
    save_results(results, args.out)
