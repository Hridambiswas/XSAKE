"""
Optimizer construction for XSAKE: AdamW + cosine LR schedule + gradient clipping.

Schedule:
  0 → warmup_steps  : linear warmup from 0 → peak_lr
  warmup → max_steps: cosine decay from peak_lr → min_lr (10% of peak)

Weight decay is applied to all parameters except LayerNorm, biases,
and embedding tables (standard practice for transformer training).
"""

from __future__ import annotations
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

from model.config import TrainingConfig


# ─── LR Schedule ─────────────────────────────────────────────────────────────

def cosine_warmup_schedule(
    peak_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> optax.Schedule:
    """
    Linear warmup → cosine decay.

    Args:
        peak_lr:       maximum learning rate after warmup
        warmup_steps:  steps for linear warmup
        total_steps:   total training steps
        min_lr_ratio:  final LR as fraction of peak_lr

    Returns:
        optax.Schedule (callable: step → lr)
    """
    min_lr = peak_lr * min_lr_ratio
    warmup = optax.linear_schedule(init_value=0.0, end_value=peak_lr, transition_steps=warmup_steps)
    decay  = optax.cosine_decay_schedule(
        init_value=peak_lr,
        decay_steps=max(1, total_steps - warmup_steps),
        alpha=min_lr_ratio,
    )
    return optax.join_schedules([warmup, decay], boundaries=[warmup_steps])


# ─── Mask for weight decay ────────────────────────────────────────────────────

def _no_decay_mask(params: Any) -> Any:
    """
    Return True for parameters that should NOT receive weight decay:
    biases, LayerNorm gamma/beta, embedding tables.
    """
    def _is_no_decay(path: tuple, _: Any) -> bool:
        key = "/".join(str(k) for k in path).lower()
        return any(s in key for s in ("bias", "gamma", "beta", "embedding"))

    return jax.tree_util.tree_map_with_path(
        lambda path, leaf: _is_no_decay(path, leaf),
        params,
    )


def make_optimizer(config: TrainingConfig) -> optax.GradientTransformation:
    """
    Build the XSAKE optimizer chain:
      clip_by_global_norm → adamw (with weight decay mask) → lr schedule

    Args:
        config: TrainingConfig

    Returns:
        optax.GradientTransformation
    """
    schedule = cosine_warmup_schedule(
        peak_lr=config.learning_rate,
        warmup_steps=config.warmup_steps,
        total_steps=config.max_steps,
    )

    return optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=config.weight_decay,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            mask=_no_decay_mask,
        ),
    )


# ─── Training State ───────────────────────────────────────────────────────────

class TrainState(NamedTuple):
    step:       jnp.ndarray          # scalar int32
    params:     Any                  # Flax param pytree
    opt_state:  optax.OptState
    rng:        jax.Array


def init_train_state(
    params: Any,
    optimizer: optax.GradientTransformation,
    rng: jax.Array,
) -> TrainState:
    return TrainState(
        step=jnp.zeros((), dtype=jnp.int32),
        params=params,
        opt_state=optimizer.init(params),
        rng=rng,
    )
