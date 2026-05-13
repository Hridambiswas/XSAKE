"""
Block-sparse attention kernel implemented in JAX Pallas.

Architecture
────────────
Two implementations share the same interface:

  sparse_attention_reference  — pure JAX with token-level masking.
                                Numerically identical to dense attention
                                on the attended blocks. Used on CPU and
                                for correctness validation.

  sparse_attention_pallas     — hand-written Pallas kernel.
                                Grid: (batch, heads, num_q_blocks).
                                Inner loop over kv_blocks; unattended blocks
                                are skipped via conditional accumulation in
                                online softmax, avoiding materialising the
                                full O(seq²) attention matrix.

  sparse_attention            — unified dispatch: Pallas on GPU/TPU,
                                reference on CPU or when use_pallas=False.

HADS integration
────────────────
block_mask may be:
  [num_q_blocks, num_kv_blocks]           — shared across all heads/batches
  [heads, num_q_blocks, num_kv_blocks]    — per-head (HADS output)
  [batch, heads, num_q_blocks, num_kv_blocks] — fully specified

The Pallas kernel expects the fully specified 4D form; broadcast is handled
by sparse_attention() before dispatch.
"""

from __future__ import annotations
import functools
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


# ─── Reference Implementation ────────────────────────────────────────────────

def sparse_attention_reference(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    block_mask: jnp.ndarray,
    block_size: int = 128,
    scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Pure JAX sparse attention. Correct but O(seq²) memory.

    Args:
        q, k, v:    [batch, heads, seq, d_head]
        block_mask: see module docstring for accepted shapes
        block_size: token block size (must divide seq)
        scale:      softmax scale (default 1/sqrt(d_head))
        causal:     apply causal mask within attended blocks

    Returns:
        output [batch, heads, seq, d_head],
        attn_weights [batch, heads, seq, seq]
    """
    batch, heads, seq, d = q.shape
    scale = scale or float(d ** -0.5)
    nb    = seq // block_size

    # Normalise block_mask → [heads, nb, nb]
    bm = _broadcast_mask(block_mask, batch=1, heads=heads, nb=nb)[0]  # [heads, nb, nb]

    # Expand block mask to token level [heads, seq, seq]
    token_mask = jnp.repeat(jnp.repeat(bm, block_size, axis=-2), block_size, axis=-1)
    token_mask = token_mask[:, :seq, :seq]

    if causal:
        causal_m = jnp.tril(jnp.ones((seq, seq), dtype=bool))
        token_mask = token_mask & causal_m[None]

    # Broadcast over batch
    token_mask = token_mask[None]   # [1, heads, seq, seq]

    scores      = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
    scores      = jnp.where(token_mask, scores, jnp.finfo(scores.dtype).min)
    attn_weights = jax.nn.softmax(scores, axis=-1)
    output      = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)

    return output, attn_weights


# ─── Pallas Kernel Body ───────────────────────────────────────────────────────

def _sparse_attn_fwd_kernel(
    q_ref,          # [1, 1, block_q, d]
    k_ref,          # [1, 1, seq_len, d]  — full sequence per batch/head
    v_ref,          # [1, 1, seq_len, d]
    bm_ref,         # [1, 1, 1, num_kv_blocks]
    o_ref,          # [1, 1, block_q, d]
    lse_ref,        # [1, 1, block_q]
    *,
    block_k: int,
    num_kv_blocks: int,
    sm_scale: float,
):
    """
    Pallas kernel: FlashAttention-style online softmax with block sparsity.

    One invocation = one (batch, head, q_block) tile.
    Iterates over kv_blocks; conditionally accumulates only attended blocks.
    """
    q  = q_ref[0, 0, :, :]       # [Bq, d]
    bm = bm_ref[0, 0, 0, :]      # [num_kv_blocks] bool
    Bq, d = q.shape

    # Online softmax state
    m_i = jnp.full((Bq,), -jnp.inf, dtype=jnp.float32)
    l_i = jnp.zeros((Bq,),          dtype=jnp.float32)
    o_i = jnp.zeros((Bq, d),        dtype=jnp.float32)

    def kv_loop(kv_idx, carry):
        m, l, o = carry
        attend = bm[kv_idx]

        # Dynamic load of the kv block
        k_blk = pl.load(
            k_ref,
            (pl.dslice(0, 1), pl.dslice(0, 1),
             pl.dslice(kv_idx * block_k, block_k), slice(None)),
        )[0, 0]   # [Bkv, d]
        v_blk = pl.load(
            v_ref,
            (pl.dslice(0, 1), pl.dslice(0, 1),
             pl.dslice(kv_idx * block_k, block_k), slice(None)),
        )[0, 0]   # [Bkv, d]

        s = jnp.dot(q.astype(jnp.float32), k_blk.astype(jnp.float32).T) * sm_scale

        m_new  = jnp.maximum(m, s.max(axis=-1))
        alpha  = jnp.exp(m - m_new)
        e_s    = jnp.exp(s - m_new[:, None])
        l_new  = alpha * l + e_s.sum(axis=-1)
        o_new  = alpha[:, None] * o + jnp.dot(e_s, v_blk.astype(jnp.float32))

        # Conditionally update — skipped blocks leave accumulators unchanged
        m = jnp.where(attend, m_new, m)
        l = jnp.where(attend, l_new, l)
        o = jnp.where(attend, o_new, o)
        return m, l, o

    m_f, l_f, o_f = jax.lax.fori_loop(0, num_kv_blocks, kv_loop, (m_i, l_i, o_i))

    o_ref[0, 0, :, :] = (o_f / l_f[:, None]).astype(q_ref.dtype)
    lse_ref[0, 0, :]  = (m_f + jnp.log(l_f)).astype(jnp.float32)


# ─── Pallas Dispatch ─────────────────────────────────────────────────────────

@functools.partial(jax.jit, static_argnames=("block_q", "block_k", "sm_scale", "causal"))
def sparse_attention_pallas(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    block_mask: jnp.ndarray,
    block_q: int = 128,
    block_k: int = 128,
    sm_scale: Optional[float] = None,
    causal: bool = False,
) -> jnp.ndarray:
    """
    Pallas-backed sparse attention.

    Requires GPU or TPU backend. Falls back gracefully via sparse_attention().
    block_mask must be broadcastable to [batch, heads, num_q_blocks, num_kv_blocks].
    """
    batch, heads, seq_len, d = q.shape
    sm_scale     = sm_scale or float(d ** -0.5)
    num_q_blocks  = seq_len // block_q
    num_kv_blocks = seq_len // block_k

    bm = _broadcast_mask(block_mask, batch=batch, heads=heads, nb=num_q_blocks)
    # bm: [batch, heads, num_q_blocks, num_kv_blocks]

    out_shape = jax.ShapeDtypeStruct(q.shape, q.dtype)
    lse_shape  = jax.ShapeDtypeStruct((batch, heads, num_q_blocks, block_q), jnp.float32)

    kernel = functools.partial(
        _sparse_attn_fwd_kernel,
        block_k=block_k,
        num_kv_blocks=num_kv_blocks,
        sm_scale=sm_scale,
    )

    # Grid: (batch, heads, num_q_blocks)
    in_specs = [
        pl.BlockSpec(lambda b, h, qi: (b, h, qi, 0), (1, 1, block_q, d)),
        pl.BlockSpec(lambda b, h, qi: (b, h, 0,  0), (1, 1, seq_len, d)),
        pl.BlockSpec(lambda b, h, qi: (b, h, 0,  0), (1, 1, seq_len, d)),
        pl.BlockSpec(lambda b, h, qi: (b, h, qi, 0), (1, 1, 1, num_kv_blocks)),
    ]
    out_specs = [
        pl.BlockSpec(lambda b, h, qi: (b, h, qi, 0), (1, 1, block_q, d)),
        pl.BlockSpec(lambda b, h, qi: (b, h, qi, 0), (1, 1, 1, block_q)),
    ]

    q4  = q.reshape(batch, heads, num_q_blocks, block_q, d)   # won't work as-is —
    # Pallas needs the arrays shaped to match the BlockSpec's world view.
    # Reshape: introduce the q-block axis so the index_map's qi maps to block index.
    #
    # The correct way: pass q as [batch, heads, num_q_blocks, block_q, d] but
    # BlockSpec for 5-D with grid (b, h, qi) → index (b, h, qi, 0, 0), shape (1, 1, 1, block_q, d).
    # For K/V: [batch, heads, 1, seq_len, d] with index (b, h, 0, 0, 0), shape (1, 1, 1, seq_len, d).

    q5  = q.reshape(batch, heads, num_q_blocks, block_q, d)
    k5  = k[:, :, None, :, :]                                 # [batch, heads, 1, seq, d]
    v5  = v[:, :, None, :, :]
    bm5 = bm[:, :, :, None, :]                               # [batch, heads, num_q_blocks, 1, num_kv_blocks]

    in_specs5 = [
        pl.BlockSpec(lambda b, h, qi: (b, h, qi, 0, 0), (1, 1, 1, block_q, d)),
        pl.BlockSpec(lambda b, h, qi: (b, h, 0,  0, 0), (1, 1, 1, seq_len, d)),
        pl.BlockSpec(lambda b, h, qi: (b, h, 0,  0, 0), (1, 1, 1, seq_len, d)),
        pl.BlockSpec(lambda b, h, qi: (b, h, qi, 0, 0), (1, 1, 1, 1, num_kv_blocks)),
    ]
    out_specs5 = [
        pl.BlockSpec(lambda b, h, qi: (b, h, qi, 0, 0), (1, 1, 1, block_q, d)),
        pl.BlockSpec(lambda b, h, qi: (b, h, qi, 0, 0), (1, 1, 1, 1, block_q)),
    ]

    def _kernel5(q_ref, k_ref, v_ref, bm_ref, o_ref, lse_ref):
        # Squeeze the extra leading dims before calling our kernel
        _sparse_attn_fwd_kernel(
            q_ref[0], k_ref[0], v_ref[0], bm_ref[0], o_ref[0], lse_ref[0],
            block_k=block_k,
            num_kv_blocks=num_kv_blocks,
            sm_scale=sm_scale,
        )

    o5, lse5 = pl.pallas_call(
        _kernel5,
        out_shape=[
            jax.ShapeDtypeStruct(q5.shape, q.dtype),
            jax.ShapeDtypeStruct((batch, heads, num_q_blocks, 1, block_q), jnp.float32),
        ],
        grid=(batch, heads, num_q_blocks),
        in_specs=in_specs5,
        out_specs=out_specs5,
    )(q5, k5, v5, bm5)

    return o5.reshape(batch, heads, seq_len, d)


# ─── Unified Entry Point ──────────────────────────────────────────────────────

def sparse_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    block_mask: jnp.ndarray,
    block_size: int = 128,
    scale: Optional[float] = None,
    causal: bool = False,
    use_pallas: bool = True,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Unified sparse attention: dispatches to Pallas on GPU/TPU, reference on CPU.

    Returns:
        (output [batch, heads, seq, d], attn_weights or None)
    """
    backend = jax.default_backend()
    if use_pallas and backend in ("gpu", "tpu"):
        out = sparse_attention_pallas(
            q, k, v, block_mask,
            block_q=block_size, block_k=block_size,
            sm_scale=scale, causal=causal,
        )
        return out, None
    return sparse_attention_reference(q, k, v, block_mask, block_size, scale, causal)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _broadcast_mask(
    block_mask: jnp.ndarray,
    batch: int,
    heads: int,
    nb: int,
) -> jnp.ndarray:
    """Broadcast block_mask to [batch, heads, nb, nb]."""
    bm = block_mask
    if bm.ndim == 2:
        bm = bm[None, None]
    elif bm.ndim == 3:
        bm = bm[None]
    return jnp.broadcast_to(bm, (batch, heads, nb, bm.shape[-1]))
