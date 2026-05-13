"""
HADS — Head-Adaptive Dynamic Sparsity

Core insight: attention heads in a transformer are not uniform.
  - Syntactic heads attend to a few fixed positions (low entropy) → high sparsity safe
  - Semantic heads attend broadly across context (high entropy) → need dense patterns

Algorithm:
  1. Calibration forward pass: collect softmax attention weights per head
  2. Compute Shannon entropy H_h for each head h
  3. Map entropy → per-head sparsity ratio (high entropy → low sparsity)
  4. Construct per-head block masks: local window always on + random long-range

This is the novel contribution of XSAKE: rather than applying one sparsity
pattern uniformly, each head gets its own pattern calibrated to its behaviour.
"""

from __future__ import annotations
from typing import NamedTuple, List, Optional
import numpy as np
import jax
import jax.numpy as jnp


# ─── Data Structures ─────────────────────────────────────────────────────────

class HADSProfile(NamedTuple):
    """Calibrated per-head sparsity configuration."""
    head_entropies: jnp.ndarray   # [n_heads]  mean entropy across calibration steps
    head_sparsity:  jnp.ndarray   # [n_heads]  assigned sparsity ratio per head
    block_masks:    jnp.ndarray   # [n_heads, num_q_blocks, num_kv_blocks] bool


# ─── Entropy Computation ─────────────────────────────────────────────────────

def compute_attention_entropy(attn_weights: jnp.ndarray) -> jnp.ndarray:
    """
    Shannon entropy of attention distributions, averaged over batch and seq positions.

    Args:
        attn_weights: [batch, heads, seq_q, seq_k] — softmax output, sums to 1 on last dim

    Returns:
        [heads] mean entropy per head
    """
    eps = 1e-9
    attn = jnp.clip(attn_weights, eps, 1.0)
    # H = -sum(p log p) over kv positions → [batch, heads, seq_q]
    H = -jnp.sum(attn * jnp.log(attn), axis=-1)
    return jnp.mean(H, axis=(0, 2))   # average over batch and q positions → [heads]


# ─── Entropy → Sparsity Mapping ──────────────────────────────────────────────

def entropy_to_sparsity(
    entropies: jnp.ndarray,
    min_sparsity: float = 0.10,
    max_sparsity: float = 0.90,
) -> jnp.ndarray:
    """
    Linear mapping: high entropy head → low sparsity, low entropy → high sparsity.

    Normalises entropy to [0, 1] then inverts so diffuse heads stay dense.

    Returns:
        [n_heads] sparsity ratios in [min_sparsity, max_sparsity]
    """
    e_min = jnp.min(entropies)
    e_max = jnp.max(entropies)
    e_range = jnp.maximum(e_max - e_min, 1e-6)

    normalized = (entropies - e_min) / e_range          # 0 = focused, 1 = diffuse
    sparsity = max_sparsity - normalized * (max_sparsity - min_sparsity)
    return sparsity


# ─── Block Mask Construction ─────────────────────────────────────────────────

def _build_head_mask(
    sparsity_ratio: float,
    num_blocks: int,
    window_size: int,
    seed: int,
    causal: bool = False,
) -> np.ndarray:
    """
    Build one head's block mask:
      - Always-on local window (window_size blocks either side)
      - Random long-range blocks to fill (1 - sparsity_ratio) budget
    """
    mask = np.zeros((num_blocks, num_blocks), dtype=bool)
    rng  = np.random.default_rng(seed)

    long_range_keep = max(1, int(num_blocks * (1.0 - sparsity_ratio)))

    for i in range(num_blocks):
        # Local window — always attend regardless of sparsity
        lo = max(0, i - window_size)
        hi = min(num_blocks, i + window_size + 1)
        mask[i, lo:hi] = True

        # Random long-range
        candidates = [j for j in range(num_blocks) if not mask[i, j]]
        if candidates:
            n = min(long_range_keep, len(candidates))
            chosen = rng.choice(candidates, size=n, replace=False)
            mask[i, chosen] = True

    if causal:
        for i in range(num_blocks):
            mask[i, i + 1 :] = False

    return mask


# ─── Profile Builder ─────────────────────────────────────────────────────────

def build_hads_profile(
    calibration_attentions: List[jnp.ndarray],
    seq_len: int,
    block_size: int,
    min_sparsity: float = 0.10,
    max_sparsity: float = 0.90,
    window_size: int = 2,
    causal: bool = False,
) -> HADSProfile:
    """
    Build a HADS sparsity profile from calibration attention weights.

    Args:
        calibration_attentions: list of [batch, heads, seq, seq] attention weight tensors
                                collected during the calibration forward passes
        seq_len:      model sequence length
        block_size:   Pallas block size
        min_sparsity: densest any head will be (high-entropy heads)
        max_sparsity: sparsest any head will be (low-entropy heads)
        window_size:  local context window (in blocks) always kept
        causal:       apply causal masking

    Returns:
        HADSProfile with per-head entropies, sparsity ratios, and block masks
    """
    # Aggregate entropy across calibration steps
    per_step = jnp.stack([compute_attention_entropy(a) for a in calibration_attentions])
    head_entropies = jnp.mean(per_step, axis=0)   # [heads]
    head_sparsity  = entropy_to_sparsity(head_entropies, min_sparsity, max_sparsity)

    n_heads   = int(head_entropies.shape[0])
    num_blocks = seq_len // block_size

    masks = np.stack([
        _build_head_mask(
            sparsity_ratio=float(head_sparsity[h]),
            num_blocks=num_blocks,
            window_size=window_size,
            seed=h,
            causal=causal,
        )
        for h in range(n_heads)
    ])   # [n_heads, num_blocks, num_blocks]

    return HADSProfile(
        head_entropies=head_entropies,
        head_sparsity=head_sparsity,
        block_masks=jnp.array(masks),
    )


def dummy_hads_profile(
    n_heads: int,
    seq_len: int,
    block_size: int,
    min_sparsity: float = 0.10,
    max_sparsity: float = 0.90,
    window_size: int = 2,
    causal: bool = False,
) -> HADSProfile:
    """
    Generate a HADS profile without calibration data.

    Assigns linearly spaced synthetic entropy values so that:
      head 0 → most focused (highest sparsity)
      head n-1 → most diffuse (lowest sparsity)

    Used for initialisation before calibration runs and for unit tests.
    """
    head_entropies = jnp.linspace(0.3, 2.5, n_heads)
    head_sparsity  = entropy_to_sparsity(head_entropies, min_sparsity, max_sparsity)
    num_blocks     = seq_len // block_size

    masks = np.stack([
        _build_head_mask(
            sparsity_ratio=float(head_sparsity[h]),
            num_blocks=num_blocks,
            window_size=window_size,
            seed=h,
            causal=causal,
        )
        for h in range(n_heads)
    ])

    return HADSProfile(
        head_entropies=head_entropies,
        head_sparsity=head_sparsity,
        block_masks=jnp.array(masks),
    )


# ─── Profile Stats ───────────────────────────────────────────────────────────

def profile_summary(profile: HADSProfile) -> dict:
    """Human-readable summary of a HADS profile."""
    sparsities = np.array(profile.head_sparsity)
    active_frac = np.array(profile.block_masks).mean(axis=(-2, -1))
    return {
        "n_heads":           int(profile.head_entropies.shape[0]),
        "mean_entropy":      float(profile.head_entropies.mean()),
        "mean_sparsity":     float(sparsities.mean()),
        "min_sparsity":      float(sparsities.min()),
        "max_sparsity":      float(sparsities.max()),
        "mean_active_frac":  float(active_frac.mean()),
        "overall_flop_ratio": float(active_frac.mean()),
    }
