"""
Per-layer gradient norm monitoring for XSAKE.

Tracks gradient norms by layer during training to detect:
  - Vanishing gradients in early layers (norms → 0)
  - Exploding gradients (norms spike before clipping)
  - Layer imbalance (some layers much larger than others)

Results are logged to W&B and stored in the metrics store.
"""

from __future__ import annotations
from typing import Any

import jax
import jax.numpy as jnp


def compute_layer_grad_norms(grads: Any) -> dict[str, float]:
    """
    Compute L2 gradient norm for each named parameter group.

    Args:
        grads: Flax gradient pytree (nested dict matching param structure)

    Returns:
        dict mapping parameter path → L2 norm
    """
    norms = {}

    def _traverse(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, val in node.items():
                _traverse(val, f"{path}/{key}" if path else key)
        elif hasattr(node, "shape"):
            norms[path] = float(jnp.sqrt(jnp.sum(node ** 2)))

    _traverse(grads, "")
    return norms


def global_grad_norm(grads: Any) -> float:
    """L2 norm of the entire gradient pytree."""
    leaves = jax.tree_util.tree_leaves(grads)
    sq_sum = sum(float(jnp.sum(g ** 2)) for g in leaves)
    return sq_sum ** 0.5


def log_gradient_stats(
    grads: Any,
    step: int,
    use_wandb: bool = False,
) -> dict:
    """
    Compute and optionally log gradient statistics.

    Returns:
        dict with global_norm, layer norms, and anomaly flags
    """
    layer_norms = compute_layer_grad_norms(grads)
    g_norm = global_grad_norm(grads)

    norms_list = list(layer_norms.values())
    stats = {
        "step":          step,
        "global_norm":   round(g_norm, 6),
        "max_layer":     round(max(norms_list), 6) if norms_list else 0.0,
        "min_layer":     round(min(norms_list), 6) if norms_list else 0.0,
        "n_zero_layers": sum(1 for n in norms_list if n < 1e-8),
        "layer_norms":   {k: round(v, 6) for k, v in layer_norms.items()},
    }

    if use_wandb:
        import wandb
        wandb.log(
            {"grad/" + k: v for k, v in layer_norms.items()} |
            {"grad/global_norm": g_norm},
            step=step,
        )

    return stats
