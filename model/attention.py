"""
XSAKE Multi-Head Sparse Attention module.

Wires the Pallas sparse attention kernel (kernels/pallas/sparse_attention.py)
and HADS sparsity profiles (kernels/hads/hads_pattern.py) into a Flax nn.Module
that is a drop-in replacement for standard multi-head attention.

Forward pass:
  1. Project Q, K, V from d_model → n_heads × d_head
  2. If a HADSProfile is attached, use per-head block masks;
     otherwise fall back to a dense or config-specified mask
  3. Call sparse_attention() — dispatches to Pallas on GPU, reference on CPU
  4. Project output back to d_model

During calibration (collect_attn=True), attention weights are returned so
that hads_pattern.build_hads_profile() can compute head entropy.
"""

from __future__ import annotations
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from model.config import ModelConfig
from kernels.pallas.sparse_attention import sparse_attention
from kernels.pallas.sparsity_mask import make_block_mask
from kernels.hads.hads_pattern import HADSProfile, dummy_hads_profile


class XSAKEAttention(nn.Module):
    """
    Multi-head sparse attention with HADS block masks.

    Attributes:
        config:      ModelConfig (vocab_size, n_heads, d_model, etc.)
        hads_profile: Optional[HADSProfile] — if None, uses config.kernel.sparsity_type
    """
    config: ModelConfig
    hads_profile: Optional[HADSProfile] = None

    def setup(self):
        cfg   = self.config
        dtype = jnp.bfloat16 if cfg.dtype == "bfloat16" else jnp.float32

        self.q_proj = nn.Dense(cfg.d_model, use_bias=cfg.bias, dtype=dtype,
                               kernel_init=nn.initializers.normal(0.02))
        self.k_proj = nn.Dense(cfg.d_model, use_bias=cfg.bias, dtype=dtype,
                               kernel_init=nn.initializers.normal(0.02))
        self.v_proj = nn.Dense(cfg.d_model, use_bias=cfg.bias, dtype=dtype,
                               kernel_init=nn.initializers.normal(0.02))
        self.out_proj = nn.Dense(cfg.d_model, use_bias=cfg.bias, dtype=dtype,
                                 kernel_init=nn.initializers.normal(0.02 / (2 * cfg.n_layers) ** 0.5))
        self.drop = nn.Dropout(rate=cfg.dropout)

    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        training: bool = False,
        collect_attn: bool = False,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Args:
            x:            [batch, seq, d_model]
            mask:         optional causal or padding token mask [batch, seq] bool
            training:     enables dropout
            collect_attn: if True returns attention weights for HADS calibration

        Returns:
            (output [batch, seq, d_model], attn_weights or None)
        """
        cfg    = self.config
        batch, seq, _ = x.shape
        h, d   = cfg.n_heads, cfg.d_head

        # Project and reshape to [batch, heads, seq, d_head]
        def _project(proj, inp):
            return proj(inp).reshape(batch, seq, h, d).transpose(0, 2, 1, 3)

        q = _project(self.q_proj, x)
        k = _project(self.k_proj, x)
        v = _project(self.v_proj, x)

        block_mask = self._get_block_mask(seq)

        use_pallas = cfg.kernel.use_pallas
        out, attn_weights = sparse_attention(
            q, k, v,
            block_mask=block_mask,
            block_size=cfg.kernel.block_size_q,
            scale=float(d ** -0.5),
            causal=True,
            use_pallas=use_pallas,
        )

        # [batch, heads, seq, d_head] → [batch, seq, d_model]
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq, cfg.d_model)
        out = self.out_proj(out)
        out = self.drop(out, deterministic=not training)

        if collect_attn:
            return out, attn_weights
        return out, None

    def _get_block_mask(self, seq: int) -> jnp.ndarray:
        """
        Return block mask [n_heads, num_q_blocks, num_kv_blocks].
        Priority: HADSProfile > config sparsity_type > dense fallback.
        """
        cfg = self.config
        bs  = cfg.kernel.block_size_q

        if seq % bs != 0:
            return jnp.ones((cfg.n_heads, seq // bs + 1, seq // bs + 1), dtype=bool)

        if self.hads_profile is not None:
            return self.hads_profile.block_masks   # [n_heads, nb, nb]

        stype = cfg.kernel.sparsity_type
        if stype == "hads":
            # Uninitialised HADS: use dummy profile (linearly spaced sparsity)
            profile = dummy_hads_profile(
                n_heads=cfg.n_heads,
                seq_len=seq,
                block_size=bs,
                min_sparsity=cfg.kernel.hads.min_sparsity,
                max_sparsity=cfg.kernel.hads.max_sparsity,
                window_size=cfg.kernel.hads.window_size,
                causal=True,
            )
            return profile.block_masks

        # Shared mask across all heads for non-HADS patterns
        shared = make_block_mask(
            seq_len=seq,
            block_size=bs,
            sparsity_type=stype,
            sparsity_ratio=0.5,
            causal=True,
        )
        return jnp.broadcast_to(shared[None], (cfg.n_heads, *shared.shape))
