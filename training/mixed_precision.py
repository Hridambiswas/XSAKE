"""
bfloat16 mixed-precision training policy for XSAKE.

Strategy (standard for JAX transformer training):
  - Forward pass:  bfloat16 (all activations and weights cast on entry)
  - Gradients:     bfloat16 (accumulated in bfloat16 during backprop)
  - Master weights: float32 (stored separately, updated in float32)
  - Loss:          float32 (always, to avoid overflow)

JAX does not have an automatic mixed-precision manager like PyTorch AMP.
Instead, we cast model parameters and inputs at the boundary and keep
a float32 master copy for the optimizer step.
"""

from __future__ import annotations
from typing import Any, Tuple

import jax
import jax.numpy as jnp


def cast_to_bf16(params: Any) -> Any:
    """Cast a pytree of parameters to bfloat16."""
    return jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if x.dtype in (jnp.float32, jnp.float16) else x,
        params,
    )


def cast_to_fp32(params: Any) -> Any:
    """Cast a pytree of parameters to float32."""
    return jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x,
        params,
    )


def bf16_forward(params_fp32: Any) -> Any:
    """
    Return bfloat16 params for the forward + backward pass.
    Master weights (fp32) are kept separately in the optimizer state.
    """
    return cast_to_bf16(params_fp32)


def scale_grads(grads: Any, loss_scale: float = 1.0) -> Any:
    """Apply a static loss scale to gradients (no dynamic scaling needed for bf16)."""
    if loss_scale == 1.0:
        return grads
    return jax.tree_util.tree_map(lambda g: g / loss_scale, grads)


def check_finite(grads: Any) -> jnp.ndarray:
    """Return True if all gradients are finite (no NaN or Inf)."""
    leaves = jax.tree_util.tree_leaves(grads)
    return jnp.all(jnp.array([jnp.all(jnp.isfinite(g)) for g in leaves]))


class MixedPrecisionPolicy:
    """
    Stateless mixed-precision helper.

    Usage in training loop:
        policy = MixedPrecisionPolicy(use_bf16=True)
        params_bf16 = policy.cast_params(params_fp32)
        # forward + backward with params_bf16
        grads_fp32  = policy.cast_grads(grads_bf16)
        # optimizer step with grads_fp32 and params_fp32
    """

    def __init__(self, use_bf16: bool = True):
        self.use_bf16 = use_bf16

    def cast_params(self, params: Any) -> Any:
        return cast_to_bf16(params) if self.use_bf16 else params

    def cast_grads(self, grads: Any) -> Any:
        return cast_to_fp32(grads) if self.use_bf16 else grads

    def cast_inputs(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.use_bf16 and x.dtype == jnp.float32:
            return x.astype(jnp.bfloat16)
        return x
