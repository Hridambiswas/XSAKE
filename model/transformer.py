"""
GPT-style transformer built on the XSAKE sparse attention kernel.

Architecture:
  TransformerBlock  — pre-norm, XSAKE attention, fused LayerNorm, MLP
  XSAKETransformer  — embedding + N blocks + head + loss

The model is parameterised by ModelConfig and accepts an optional
HADSProfile that gets distributed to every attention layer.

Gradient checkpointing is supported via nn.remat on TransformerBlock,
halving activation memory at the cost of one extra forward pass per block.
"""

from __future__ import annotations
from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from model.config import ModelConfig
from model.embedding import TransformerEmbedding
from model.attention import XSAKEAttention
from kernels.pallas.fused_layernorm import fused_layernorm
from kernels.hads.hads_pattern import HADSProfile


# ─── MLP ─────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        cfg   = self.config
        dtype = jnp.bfloat16 if cfg.dtype == "bfloat16" else jnp.float32
        x = nn.Dense(cfg.d_ff,    use_bias=cfg.bias, dtype=dtype,
                     kernel_init=nn.initializers.normal(0.02))(x)
        x = nn.gelu(x)
        x = nn.Dense(cfg.d_model, use_bias=cfg.bias, dtype=dtype,
                     kernel_init=nn.initializers.normal(0.02 / (2 * cfg.n_layers) ** 0.5))(x)
        x = nn.Dropout(rate=cfg.dropout, deterministic=not training)(x)
        return x


# ─── LayerNorm wrapper ────────────────────────────────────────────────────────

class LayerNorm(nn.Module):
    d_model: int
    eps: float = 1e-5
    use_pallas: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        dtype  = x.dtype
        gamma  = self.param("gamma", nn.initializers.ones,  (self.d_model,))
        beta   = self.param("beta",  nn.initializers.zeros, (self.d_model,))
        return fused_layernorm(
            x.astype(jnp.float32),
            gamma.astype(jnp.float32),
            beta.astype(jnp.float32),
            eps=self.eps,
            use_pallas=self.use_pallas,
        ).astype(dtype)


# ─── Transformer Block ────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    config: ModelConfig
    hads_profile: Optional[HADSProfile] = None

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False,
        collect_attn: bool = False,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        cfg = self.config

        # Pre-norm attention
        residual = x
        x = LayerNorm(cfg.d_model, use_pallas=cfg.kernel.use_pallas)(x)
        x, attn_weights = XSAKEAttention(
            config=cfg,
            hads_profile=self.hads_profile,
        )(x, training=training, collect_attn=collect_attn)
        x = residual + x

        # Pre-norm MLP
        residual = x
        x = LayerNorm(cfg.d_model, use_pallas=cfg.kernel.use_pallas)(x)
        x = MLP(cfg)(x, training=training)
        x = residual + x

        return x, attn_weights


# ─── Full Transformer ─────────────────────────────────────────────────────────

class XSAKETransformer(nn.Module):
    """
    GPT-style autoregressive transformer using XSAKE sparse attention.

    Args:
        config:       ModelConfig
        hads_profile: Optional HADSProfile — if provided, shared across all layers
        grad_checkpoint: wrap each block with nn.remat for activation checkpointing
    """
    config: ModelConfig
    hads_profile: Optional[HADSProfile] = None
    grad_checkpoint: bool = False

    @nn.compact
    def __call__(
        self,
        token_ids: jnp.ndarray,
        training: bool = False,
        collect_attn: bool = False,
    ) -> Tuple[jnp.ndarray, Optional[list]]:
        """
        Args:
            token_ids:    [batch, seq] int32
            training:     enables dropout
            collect_attn: collect attention weights from all layers (for HADS calibration)

        Returns:
            (logits [batch, seq, vocab_size], layer_attn_weights or None)
        """
        cfg   = self.config
        dtype = jnp.bfloat16 if cfg.dtype == "bfloat16" else jnp.float32

        x = TransformerEmbedding(cfg)(token_ids, training=training)

        all_attn = [] if collect_attn else None
        block_fn = TransformerBlock

        if self.grad_checkpoint:
            block_fn = nn.remat(TransformerBlock, prevent_cse=False)

        for layer_idx in range(cfg.n_layers):
            x, attn = block_fn(
                config=cfg,
                hads_profile=self.hads_profile,
                name=f"block_{layer_idx}",
            )(x, training=training, collect_attn=collect_attn)

            if collect_attn and attn is not None:
                all_attn.append(attn)

        x = LayerNorm(cfg.d_model, use_pallas=cfg.kernel.use_pallas)(x)

        # Tie output projection weights to token embedding (weight tying)
        embed_table = self.variables.get("params", {}).get(
            "TransformerEmbedding_0", {}
        ).get("TokenEmbedding_0", {}).get("Embed_0", {}).get("embedding", None)

        if embed_table is not None:
            logits = x.astype(jnp.float32) @ embed_table.T.astype(jnp.float32)
        else:
            logits = nn.Dense(
                cfg.vocab_size,
                use_bias=False,
                dtype=jnp.float32,
                kernel_init=nn.initializers.normal(0.02),
                name="lm_head",
            )(x.astype(jnp.float32))

        return logits, all_attn


# ─── Loss ────────────────────────────────────────────────────────────────────

def cross_entropy_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    ignore_index: int = -1,
) -> jnp.ndarray:
    """
    Cross-entropy language model loss with optional padding mask.

    Args:
        logits: [batch, seq, vocab_size]
        labels: [batch, seq] int32 — next-token targets; -1 positions are ignored
        ignore_index: token id to mask out of loss (typically padding)

    Returns:
        scalar mean loss
    """
    vocab = logits.shape[-1]
    valid = labels != ignore_index

    # Shift: predict token t+1 from position t
    flat_logits = logits[:, :-1].reshape(-1, vocab)
    flat_labels = labels[:, 1:].reshape(-1)
    flat_valid  = valid[:, 1:].reshape(-1)

    loss = jax.vmap(
        lambda lg, lb: -jax.nn.log_softmax(lg, axis=-1)[lb]
    )(flat_logits, jnp.clip(flat_labels, 0, vocab - 1))

    return jnp.sum(loss * flat_valid) / jnp.maximum(flat_valid.sum(), 1)
