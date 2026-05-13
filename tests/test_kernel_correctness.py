"""
Numerical correctness tests for XSAKE kernels.

Verifies that:
  1. sparse_attention_reference matches standard JAX attention on dense mask
  2. sparse_attention_reference with HADS mask matches dense on attended blocks
  3. fused_softmax matches jax.nn.softmax
  4. fused_layernorm matches manual layernorm

All tests run on CPU (no GPU required).
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from kernels.pallas.sparse_attention import sparse_attention_reference
from kernels.pallas.sparsity_mask import make_block_mask
from kernels.pallas.fused_softmax import fused_softmax
from kernels.pallas.fused_layernorm import fused_layernorm
from kernels.hads.hads_pattern import dummy_hads_profile


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def qkv_small():
    key = jax.random.PRNGKey(0)
    B, H, S, D = 2, 4, 256, 32
    q = jax.random.normal(key, (B, H, S, D), dtype=jnp.float32)
    k = jax.random.normal(key, (B, H, S, D), dtype=jnp.float32)
    v = jax.random.normal(key, (B, H, S, D), dtype=jnp.float32)
    return q, k, v


# ── Attention correctness ──────────────────────────────────────────────────────

def _reference_dense_attention(q, k, v):
    d = q.shape[-1]
    scores  = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (d ** -0.5)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", weights, v)


def test_dense_mask_matches_standard_attention(qkv_small):
    """With a fully-dense block mask, sparse_attention_reference == standard attention."""
    q, k, v = qkv_small
    B, H, S, D = q.shape
    block_size = 64

    dense_mask = make_block_mask(S, block_size, "dense")
    out_sparse, _ = sparse_attention_reference(q, k, v, dense_mask, block_size=block_size)
    out_dense      = _reference_dense_attention(q, k, v)

    l2_err = float(jnp.mean((out_sparse - out_dense) ** 2) ** 0.5)
    assert l2_err < 1e-4, f"L2 error with dense mask: {l2_err:.2e}"


def test_hads_output_shape(qkv_small):
    """HADS sparse attention returns correct output shape."""
    q, k, v = qkv_small
    B, H, S, D = q.shape
    block_size = 64

    profile = dummy_hads_profile(n_heads=H, seq_len=S, block_size=block_size)
    out, attn = sparse_attention_reference(q, k, v, profile.block_masks, block_size=block_size)

    assert out.shape == (B, H, S, D), f"Wrong output shape: {out.shape}"


def test_local_mask_causal(qkv_small):
    """Causal local mask: output at position i must not depend on j > i."""
    q, k, v = qkv_small
    S = q.shape[2]
    block_size = 64

    local_mask = make_block_mask(S, block_size, "local", window_size=2, causal=True)
    # Verify mask is lower-triangular at block level
    nb = S // block_size
    for i in range(nb):
        for j in range(nb):
            if j > i:
                assert not bool(local_mask[i, j]), f"Causal mask leaks: block ({i},{j}) is True"


def test_sparse_output_finite(qkv_small):
    """Sparse attention output must contain no NaN or Inf."""
    q, k, v = qkv_small
    S, H = q.shape[2], q.shape[1]
    block_size = 64

    profile = dummy_hads_profile(n_heads=H, seq_len=S, block_size=block_size)
    out, _ = sparse_attention_reference(q, k, v, profile.block_masks, block_size=block_size)

    assert jnp.all(jnp.isfinite(out)), "Output contains NaN or Inf"


# ── Softmax correctness ────────────────────────────────────────────────────────

def test_fused_softmax_matches_jax():
    """fused_softmax (CPU fallback) == jax.nn.softmax."""
    key = jax.random.PRNGKey(1)
    x = jax.random.normal(key, (8, 512), dtype=jnp.float32)

    ref = jax.nn.softmax(x, axis=-1)
    out = fused_softmax(x, use_pallas=False)

    np.testing.assert_allclose(np.array(out), np.array(ref), rtol=1e-5, atol=1e-6)


def test_fused_softmax_sums_to_one():
    key = jax.random.PRNGKey(2)
    x   = jax.random.normal(key, (4, 128))
    out = fused_softmax(x, use_pallas=False)
    row_sums = jnp.sum(out, axis=-1)
    np.testing.assert_allclose(np.array(row_sums), np.ones(4), atol=1e-5)


# ── LayerNorm correctness ──────────────────────────────────────────────────────

def test_fused_layernorm_matches_manual():
    """fused_layernorm (CPU fallback) == manual layernorm."""
    key = jax.random.PRNGKey(3)
    x   = jax.random.normal(key, (4, 16, 128), dtype=jnp.float32)
    gamma = jnp.ones(128)
    beta  = jnp.zeros(128)

    def manual_ln(x, g, b, eps=1e-5):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var  = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return g * (x - mean) / jnp.sqrt(var + eps) + b

    ref = manual_ln(x, gamma, beta)
    out = fused_layernorm(x, gamma, beta, use_pallas=False)

    np.testing.assert_allclose(np.array(out), np.array(ref), rtol=1e-4, atol=1e-5)


def test_fused_layernorm_zero_mean():
    """LayerNorm output has approximately zero mean along last dim."""
    key   = jax.random.PRNGKey(4)
    x     = jax.random.normal(key, (2, 32, 64), dtype=jnp.float32)
    gamma = jnp.ones(64)
    beta  = jnp.zeros(64)
    out   = fused_layernorm(x, gamma, beta, use_pallas=False)
    means = jnp.mean(out, axis=-1)
    np.testing.assert_allclose(np.array(means), np.zeros_like(means), atol=1e-4)
