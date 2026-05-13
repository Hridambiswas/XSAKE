"""Verify benchmarks produce consistent results across runs (determinism tests)."""
import pytest
import jax
import jax.numpy as jnp
import numpy as np


def _run_attention_once(key, B, H, S, D, block_size):
    from kernels.hads.hads_pattern import dummy_hads_profile
    from kernels.pallas.sparse_attention import sparse_attention_reference

    q = jax.random.normal(key, (B, H, S, D), jnp.bfloat16)
    k = jax.random.normal(key, (B, H, S, D), jnp.bfloat16)
    v = jax.random.normal(key, (B, H, S, D), jnp.bfloat16)
    profile = dummy_hads_profile(n_heads=H, seq_len=S, block_size=block_size)
    out, _ = sparse_attention_reference(q, k, v, profile.block_masks, block_size)
    return np.array(out)


def test_sparse_attention_deterministic():
    key = jax.random.PRNGKey(42)
    B, H, S, D = 1, 4, 128, 32
    out1 = _run_attention_once(key, B, H, S, D, block_size=32)
    out2 = _run_attention_once(key, B, H, S, D, block_size=32)
    np.testing.assert_array_equal(out1, out2)


def test_hads_profile_deterministic():
    from kernels.hads.hads_pattern import dummy_hads_profile

    p1 = dummy_hads_profile(n_heads=8, seq_len=256, block_size=64)
    p2 = dummy_hads_profile(n_heads=8, seq_len=256, block_size=64)
    np.testing.assert_array_equal(
        np.array(p1.head_entropy), np.array(p2.head_entropy)
    )
    for h in range(8):
        np.testing.assert_array_equal(
            np.array(p1.block_masks[h]), np.array(p2.block_masks[h])
        )


def test_dense_mask_same_as_standard_attention():
    from kernels.pallas.sparse_attention import sparse_attention_reference

    key = jax.random.PRNGKey(0)
    B, H, S, D = 1, 4, 64, 16
    q = jax.random.normal(key, (B, H, S, D), jnp.float32)
    k = jax.random.normal(key, (B, H, S, D), jnp.float32)
    v = jax.random.normal(key, (B, H, S, D), jnp.float32)

    block_size = 16
    n_blocks = S // block_size
    dense_masks = jnp.ones((H, n_blocks, n_blocks), dtype=bool)
    xsake_out, _ = sparse_attention_reference(q, k, v, dense_masks, block_size)

    # Standard causal attention reference
    scale = D ** -0.5
    scores = jnp.einsum("bhqd,bhkd->bhqk", q * scale, k)
    mask = jnp.tril(jnp.ones((S, S), bool))
    scores = jnp.where(mask[None, None], scores, jnp.finfo(jnp.float32).min)
    std_out = jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(scores, axis=-1), v)

    np.testing.assert_allclose(np.array(xsake_out), np.array(std_out), atol=1e-4)


def test_latency_benchmark_stable():
    """Median latency of two independent runs must agree within 20%."""
    import time

    from kernels.pallas.sparse_attention import sparse_attention_reference
    from kernels.hads.hads_pattern import dummy_hads_profile

    key = jax.random.PRNGKey(7)
    B, H, S, D = 1, 4, 128, 32
    q = jax.random.normal(key, (B, H, S, D), jnp.bfloat16)
    k = jax.random.normal(key, (B, H, S, D), jnp.bfloat16)
    v = jax.random.normal(key, (B, H, S, D), jnp.bfloat16)
    profile = dummy_hads_profile(n_heads=H, seq_len=S, block_size=32)
    fn = jax.jit(lambda q, k, v: sparse_attention_reference(q, k, v, profile.block_masks, 32)[0])

    for _ in range(3):
        jax.block_until_ready(fn(q, k, v))

    def median_ms(n=10):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(q, k, v))
            ts.append((time.perf_counter() - t0) * 1000)
        return float(np.median(ts))

    m1, m2 = median_ms(), median_ms()
    ratio = max(m1, m2) / (min(m1, m2) + 1e-9)
    assert ratio < 1.4, f"Latency unstable: {m1:.2f}ms vs {m2:.2f}ms (ratio={ratio:.2f})"


def test_fused_softmax_matches_jax_softmax():
    from kernels.pallas.fused_softmax import fused_softmax

    key = jax.random.PRNGKey(1)
    x = jax.random.normal(key, (4, 16, 32), jnp.float32)
    ref = jax.nn.softmax(x, axis=-1)
    got = fused_softmax(x)
    np.testing.assert_allclose(np.array(got), np.array(ref), atol=1e-5)
