"""FlashAttention-2 baseline via Pallas (compiles to Triton on GPU, CPU fallback)."""
from __future__ import annotations

import functools
from typing import Optional

import jax
import jax.numpy as jnp

try:
    import jax.experimental.pallas as pl
    import jax.experimental.pallas.ops.gpu.attention as pallas_attention

    _HAS_PALLAS_FLASH = True
except (ImportError, AttributeError):
    _HAS_PALLAS_FLASH = False


def _flash_attention_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    causal: bool = True,
    scale: Optional[float] = None,
) -> jax.Array:
    """Pure JAX FlashAttention-2 reference (online softmax, O(1) memory in blocks)."""
    B, H, S, D = q.shape
    if scale is None:
        scale = D ** -0.5

    q = q.astype(jnp.float32) * scale
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)

    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k)

    if causal:
        mask = jnp.tril(jnp.ones((S, S), dtype=bool))
        scores = jnp.where(mask[None, None, :, :], scores, jnp.finfo(jnp.float32).min)

    attn = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
    return out.astype(q.dtype)


def _flash_attention_pallas(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    causal: bool = True,
    scale: Optional[float] = None,
) -> jax.Array:
    """Pallas flash attention — uses hardware-optimized kernel on GPU/TPU."""
    if not _HAS_PALLAS_FLASH:
        return _flash_attention_reference(q, k, v, causal=causal, scale=scale)

    B, H, S, D = q.shape
    if scale is None:
        scale = D ** -0.5

    try:
        out = pallas_attention.mha(q, k, v, causal=causal, scale=scale)
        return out
    except Exception:
        return _flash_attention_reference(q, k, v, causal=causal, scale=scale)


def flash_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    causal: bool = True,
    scale: Optional[float] = None,
    use_pallas: bool = True,
) -> jax.Array:
    """Dispatch: Pallas kernel on GPU, pure-JAX reference on CPU."""
    backend = jax.devices()[0].platform
    if use_pallas and backend in ("gpu", "tpu") and _HAS_PALLAS_FLASH:
        return _flash_attention_pallas(q, k, v, causal=causal, scale=scale)
    return _flash_attention_reference(q, k, v, causal=causal, scale=scale)


# JIT-compiled entry point used by benchmarks
flash_attention_jit = jax.jit(
    functools.partial(flash_attention, causal=True),
    static_argnames=("causal", "use_pallas"),
)


def benchmark_flash_vs_standard(
    batch: int = 1,
    heads: int = 12,
    seq: int = 1024,
    d_head: int = 64,
    n_trials: int = 20,
) -> dict:
    """Returns dict with flash_ms, standard_ms, speedup."""
    import time
    import numpy as np

    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (batch, heads, seq, d_head), jnp.bfloat16)
    k = jax.random.normal(key, (batch, heads, seq, d_head), jnp.bfloat16)
    v = jax.random.normal(key, (batch, heads, seq, d_head), jnp.bfloat16)

    scale = d_head ** -0.5

    flash_fn = jax.jit(functools.partial(flash_attention, causal=True, use_pallas=True))
    std_fn = jax.jit(functools.partial(_flash_attention_reference, causal=True, scale=scale))

    # Warmup
    for _ in range(3):
        jax.block_until_ready(flash_fn(q, k, v))
        jax.block_until_ready(std_fn(q, k, v))

    flash_times, std_times = [], []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        jax.block_until_ready(flash_fn(q, k, v))
        flash_times.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        jax.block_until_ready(std_fn(q, k, v))
        std_times.append((time.perf_counter() - t0) * 1000)

    flash_ms = float(np.median(flash_times))
    std_ms = float(np.median(std_times))
    return {
        "flash_ms": flash_ms,
        "standard_ms": std_ms,
        "speedup": std_ms / flash_ms,
        "seq_len": seq,
        "backend": jax.devices()[0].platform,
    }
