"""
Block-level sparsity mask generation for sparse attention.

A block mask M[num_q_blocks, num_kv_blocks] is a boolean matrix where
M[i, j] = True means q_block i is allowed to attend to kv_block j.
False blocks are skipped entirely during kernel execution.

Supported patterns:
  dense         — full attention (all True), baseline
  local         — sliding window of adjacent blocks
  random        — random subset of kv blocks per q block
  bigbird       — local + global tokens + random (BigBird-style)
  hads          — head-adaptive (produced by kernels/hads/hads_pattern.py)
"""

from __future__ import annotations
from typing import Literal, Optional
import numpy as np
import jax.numpy as jnp
import jax

SparsityType = Literal["dense", "local", "random", "bigbird"]


def make_block_mask(
    seq_len: int,
    block_size: int,
    sparsity_type: SparsityType,
    sparsity_ratio: float = 0.5,
    window_size: int = 3,
    global_blocks: int = 1,
    causal: bool = False,
    rng: Optional[jax.Array] = None,
) -> jnp.ndarray:
    """
    Build a block sparsity mask.

    Args:
        seq_len:       total sequence length (must be divisible by block_size)
        block_size:    tokens per block
        sparsity_type: which pattern to generate
        sparsity_ratio: fraction of kv blocks to SKIP (0 = dense, 1 = nothing attended)
        window_size:   number of adjacent blocks for local pattern
        global_blocks: number of global blocks for bigbird pattern
        causal:        if True, zero out future kv blocks
        rng:           JAX RNG key for random patterns

    Returns:
        jnp.ndarray bool [num_q_blocks, num_kv_blocks]
    """
    assert seq_len % block_size == 0, "seq_len must be divisible by block_size"
    nb = seq_len // block_size

    if sparsity_type == "dense":
        mask = np.ones((nb, nb), dtype=bool)

    elif sparsity_type == "local":
        mask = np.zeros((nb, nb), dtype=bool)
        for i in range(nb):
            lo = max(0, i - window_size)
            hi = min(nb, i + window_size + 1)
            mask[i, lo:hi] = True

    elif sparsity_type == "random":
        keep = max(1, int(nb * (1.0 - sparsity_ratio)))
        mask = np.zeros((nb, nb), dtype=bool)
        np_rng = np.random.default_rng(
            int(jax.random.randint(rng or jax.random.PRNGKey(0), (), 0, 2**31))
        )
        for i in range(nb):
            chosen = np_rng.choice(nb, size=keep, replace=False)
            mask[i, chosen] = True

    elif sparsity_type == "bigbird":
        mask = np.zeros((nb, nb), dtype=bool)
        g = min(global_blocks, nb)
        rand_keep = max(1, int(nb * 0.15))
        np_rng = np.random.default_rng(42)

        for i in range(nb):
            # Global tokens (first g blocks)
            mask[i, :g] = True
            # Local window
            lo = max(0, i - window_size)
            hi = min(nb, i + window_size + 1)
            mask[i, lo:hi] = True
            # Random long-range
            candidates = np.where(~mask[i])[0]
            if len(candidates) > 0:
                chosen = np_rng.choice(candidates, size=min(rand_keep, len(candidates)), replace=False)
                mask[i, chosen] = True
    else:
        raise ValueError(
            f"Unknown sparsity_type '{sparsity_type}'. "
            "For 'hads', use kernels.hads.hads_pattern.build_hads_profile()."
        )

    if causal:
        for i in range(nb):
            mask[i, i + 1 :] = False

    return jnp.array(mask)


def mask_to_token_level(
    block_mask: jnp.ndarray,
    block_size: int,
    seq_len: int,
) -> jnp.ndarray:
    """
    Expand a block mask [num_q_blocks, num_kv_blocks] to token level [seq, seq].
    Useful for the reference attention implementation.
    """
    nb = seq_len // block_size
    bm = np.array(block_mask[:nb, :nb])
    token_mask = np.repeat(np.repeat(bm, block_size, axis=0), block_size, axis=1)
    return jnp.array(token_mask[:seq_len, :seq_len])


def sparsity_ratio(block_mask: jnp.ndarray) -> float:
    """Fraction of blocks that are zeroed out (skipped)."""
    total = block_mask.size
    active = int(jnp.sum(block_mask))
    return 1.0 - active / total
