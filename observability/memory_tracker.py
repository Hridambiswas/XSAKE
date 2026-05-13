"""
Per-step HBM memory usage tracking for XSAKE.

Tracks:
  - Peak device memory per training step
  - Activation memory vs parameter memory breakdown
  - Memory savings from HADS sparsity vs dense baseline

On CPU/Mac: returns zero (unified memory, no HBM).
On GPU: reads from jax device memory_stats().
"""

from __future__ import annotations
from typing import Optional

import jax


def device_memory_mb() -> float:
    """Current device memory in use (MB). Returns 0.0 on CPU."""
    backend = jax.default_backend()
    if backend not in ("gpu", "tpu"):
        return 0.0
    try:
        dev   = jax.devices()[0]
        stats = dev.memory_stats()
        return stats.get("bytes_in_use", 0) / 1e6
    except Exception:
        return 0.0


def peak_memory_mb() -> float:
    """Peak device memory allocated so far (MB). Returns 0.0 on CPU."""
    backend = jax.default_backend()
    if backend not in ("gpu", "tpu"):
        return 0.0
    try:
        dev   = jax.devices()[0]
        stats = dev.memory_stats()
        return stats.get("peak_bytes_in_use", 0) / 1e6
    except Exception:
        return 0.0


class MemoryTracker:
    """
    Context manager that measures net memory allocated by a block of code.

    Usage:
        with MemoryTracker() as mt:
            out = sparse_attention_pallas(q, k, v, mask)
            jax.block_until_ready(out)
        print(f"Memory used: {mt.delta_mb:.1f} MB")
    """

    def __init__(self):
        self.before_mb: float = 0.0
        self.after_mb:  float = 0.0
        self.delta_mb:  float = 0.0

    def __enter__(self):
        self.before_mb = device_memory_mb()
        return self

    def __exit__(self, *args):
        self.after_mb = device_memory_mb()
        self.delta_mb = max(0.0, self.after_mb - self.before_mb)


def estimate_param_memory_mb(params) -> float:
    """
    Estimate parameter memory in MB from a Flax parameter pytree.
    """
    import jax.numpy as jnp
    leaves = jax.tree_util.tree_leaves(params)
    total_bytes = sum(x.size * x.dtype.itemsize for x in leaves)
    return total_bytes / 1e6


def estimate_activation_memory_mb(
    batch: int,
    seq: int,
    d_model: int,
    n_layers: int,
    sparsity: float = 0.0,
    dtype_bytes: int = 2,
) -> float:
    """
    Approximate activation memory for gradient checkpointing analysis.

    Without checkpointing: stores all layer activations.
    With checkpointing (nn.remat): re-computes activations, stores only one layer.
    """
    per_layer = batch * seq * d_model * dtype_bytes  # residual stream
    attn_scores = batch * seq * seq * dtype_bytes * (1 - sparsity)  # attention matrix
    total = n_layers * (per_layer + attn_scores)
    return total / 1e6
