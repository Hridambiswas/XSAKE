"""
Tests for the HADS sparsity pattern algorithm.

Verifies:
  1. Block mask shape and dtype
  2. Sparsity ratio is within [min, max] bounds
  3. Entropy → sparsity mapping is monotone decreasing
  4. Local window is always present regardless of sparsity
  5. Calibration-based profile matches expected entropy ordering
"""

import pytest
import jax.numpy as jnp
import numpy as np

from kernels.hads.hads_pattern import (
    dummy_hads_profile,
    build_hads_profile,
    compute_attention_entropy,
    entropy_to_sparsity,
    profile_summary,
)
from kernels.pallas.sparsity_mask import sparsity_ratio


N_HEADS = 8
SEQ_LEN = 512
BLOCK_SIZE = 64


@pytest.fixture
def dummy_profile():
    return dummy_hads_profile(N_HEADS, SEQ_LEN, BLOCK_SIZE)


def test_block_mask_shape(dummy_profile):
    nb = SEQ_LEN // BLOCK_SIZE
    assert dummy_profile.block_masks.shape == (N_HEADS, nb, nb)


def test_block_mask_dtype(dummy_profile):
    assert dummy_profile.block_masks.dtype == jnp.bool_


def test_head_count(dummy_profile):
    assert dummy_profile.head_entropies.shape == (N_HEADS,)
    assert dummy_profile.head_sparsity.shape  == (N_HEADS,)


def test_sparsity_bounds(dummy_profile):
    """All per-head sparsity ratios must be within [0.10, 0.90]."""
    s = np.array(dummy_profile.head_sparsity)
    assert np.all(s >= 0.09), f"Sparsity below min: {s.min():.3f}"
    assert np.all(s <= 0.91), f"Sparsity above max: {s.max():.3f}"


def test_entropy_to_sparsity_monotone():
    """Higher entropy → lower sparsity (monotone decreasing)."""
    entropies = jnp.linspace(0.1, 3.0, 20)
    sparsities = entropy_to_sparsity(entropies)
    diffs = jnp.diff(sparsities)
    assert jnp.all(diffs <= 1e-6), "entropy→sparsity mapping is not monotone decreasing"


def test_local_window_always_present():
    """Self-attention block (diagonal) must always be True regardless of sparsity."""
    profile = dummy_hads_profile(N_HEADS, SEQ_LEN, BLOCK_SIZE, window_size=1)
    nb = SEQ_LEN // BLOCK_SIZE
    for h in range(N_HEADS):
        for i in range(nb):
            assert profile.block_masks[h, i, i], \
                f"Head {h}, diagonal block ({i},{i}) is False"


def test_calibration_profile():
    """Calibration-based profile: heads with higher entropy get lower sparsity."""
    import jax
    key = jax.random.PRNGKey(0)

    # Simulate attention weights: some heads focused, some diffuse
    focused_attn = jax.nn.softmax(
        jnp.eye(SEQ_LEN)[None, None].repeat(2, 0).repeat(N_HEADS // 2, 1) * 10,
        axis=-1
    )
    diffuse_attn = jax.nn.softmax(
        jax.random.normal(key, (2, N_HEADS // 2, SEQ_LEN, SEQ_LEN)),
        axis=-1
    )
    # Concatenate into [batch, heads, seq, seq]
    attn = jnp.concatenate([focused_attn, diffuse_attn], axis=1)

    profile = build_hads_profile(
        calibration_attentions=[attn],
        seq_len=SEQ_LEN,
        block_size=BLOCK_SIZE,
    )
    # Focused heads (first half) should have lower entropy → higher sparsity
    focused_sp = float(profile.head_sparsity[:N_HEADS // 2].mean())
    diffuse_sp = float(profile.head_sparsity[N_HEADS // 2:].mean())
    assert focused_sp > diffuse_sp, \
        f"Focused heads should be sparser: {focused_sp:.3f} vs {diffuse_sp:.3f}"


def test_profile_summary_keys(dummy_profile):
    summary = profile_summary(dummy_profile)
    for key in ["n_heads", "mean_entropy", "mean_sparsity", "mean_active_frac"]:
        assert key in summary, f"Missing key: {key}"
