"""
Token and positional embeddings for XSAKE transformer.

Uses learned absolute positional embeddings (GPT-2 style).
Embeddings are initialised with small normal noise and cast to the
model's training dtype (bfloat16 by default) at forward time.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import flax.linen as nn
from model.config import ModelConfig


class TokenEmbedding(nn.Module):
    vocab_size: int
    d_model: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            token_ids: [batch, seq] int32

        Returns:
            [batch, seq, d_model]
        """
        embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
        )
        return embed(token_ids)


class PositionalEmbedding(nn.Module):
    seq_len: int
    d_model: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, positions: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            positions: [batch, seq] int32 — typically jnp.arange(seq)[None]

        Returns:
            [batch, seq, d_model]
        """
        pos_embed = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.01),
            (self.seq_len, self.d_model),
        )
        return pos_embed[positions].astype(self.dtype)


class TransformerEmbedding(nn.Module):
    """Combined token + position embedding with dropout."""
    config: ModelConfig

    @nn.compact
    def __call__(
        self,
        token_ids: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        """
        Args:
            token_ids: [batch, seq] int32
            training:  enables dropout when True

        Returns:
            [batch, seq, d_model]
        """
        cfg   = self.config
        dtype = jnp.bfloat16 if cfg.dtype == "bfloat16" else jnp.float32

        batch, seq = token_ids.shape
        positions  = jnp.broadcast_to(jnp.arange(seq)[None], (batch, seq))

        tok_emb = TokenEmbedding(cfg.vocab_size, cfg.d_model, dtype)(token_ids)
        pos_emb = PositionalEmbedding(cfg.seq_len, cfg.d_model, dtype)(positions)

        x = tok_emb + pos_emb
        x = nn.Dropout(rate=cfg.dropout, deterministic=not training)(x)
        return x
