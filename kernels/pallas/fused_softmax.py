"""
Fused softmax Pallas kernel.

Implements a numerically stable online softmax fused into a single kernel pass:
  1. Compute max (for numerical stability)
  2. Compute exp(x - max) and sum
  3. Normalise

On GPU this avoids two separate global memory reads (one for max, one for exp/sum)
that a naive implementation would require. The fused kernel reads the input once.

On CPU, falls back to jax.nn.softmax.
"""

from __future__ import annotations
import functools
from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


# ─── Pallas Kernel ───────────────────────────────────────────────────────────

def _softmax_kernel(x_ref, o_ref, *, axis_size: int):
    """
    Fused max + exp + normalise in one kernel invocation.

    x_ref: [block_rows, axis_size]
    o_ref: [block_rows, axis_size]
    """
    x = x_ref[...]                              # [block_rows, axis_size]
    x_max = jnp.max(x, axis=-1, keepdims=True)  # [block_rows, 1]
    e = jnp.exp(x - x_max)
    o_ref[...] = e / jnp.sum(e, axis=-1, keepdims=True)


@functools.partial(jax.jit, static_argnames=("block_size",))
def fused_softmax_pallas(x: jnp.ndarray, block_size: int = 256) -> jnp.ndarray:
    """
    Pallas fused softmax over last dimension.

    Args:
        x:          [..., seq] — arbitrary batch dims, softmax over last axis
        block_size: number of rows processed per kernel invocation

    Returns:
        softmax(x) — same shape as x
    """
    *batch_dims, axis_size = x.shape
    flat = x.reshape(-1, axis_size)
    n_rows = flat.shape[0]

    # Pad to multiple of block_size
    pad = (block_size - n_rows % block_size) % block_size
    if pad:
        flat = jnp.concatenate([flat, jnp.zeros((pad, axis_size), dtype=flat.dtype)], axis=0)

    n_blocks = flat.shape[0] // block_size

    out = pl.pallas_call(
        functools.partial(_softmax_kernel, axis_size=axis_size),
        out_shape=jax.ShapeDtypeStruct(flat.shape, flat.dtype),
        grid=(n_blocks,),
        in_specs=[pl.BlockSpec(lambda i: (i, 0), (block_size, axis_size))],
        out_specs=pl.BlockSpec(lambda i: (i, 0), (block_size, axis_size)),
    )(flat)

    if pad:
        out = out[:n_rows]
    return out.reshape(*batch_dims, axis_size)


def fused_softmax(
    x: jnp.ndarray,
    block_size: int = 256,
    use_pallas: bool = True,
) -> jnp.ndarray:
    """
    Numerically stable softmax over last dimension.

    Dispatches to Pallas kernel on GPU/TPU, jax.nn.softmax on CPU.
    """
    if use_pallas and jax.default_backend() in ("gpu", "tpu"):
        return fused_softmax_pallas(x, block_size=block_size)
    return jax.nn.softmax(x, axis=-1)
