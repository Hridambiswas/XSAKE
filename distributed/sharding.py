"""
JAX NamedSharding specifications for XSAKE model parameters and activations.

Sharding strategy (2D mesh: data × model):
  - Embedding table:     (None, None)        — replicated
  - QKV / dense weights: (None, model_axis)  — column-sharded (model parallel)
  - Attention output:    (model_axis, None)  — row-sharded
  - LayerNorm params:    (None,)             — replicated
  - Batch / activations: (data_axis, None, None) — sharded on batch dimension

On a single device these specs are no-ops (everything is on device 0).
"""

from __future__ import annotations
from typing import Any

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


# ─── Partition Specs ─────────────────────────────────────────────────────────

def get_partition_specs(data_axis: str = "data", model_axis: str = "model") -> dict:
    """
    Return a dict mapping parameter path fragments → PartitionSpec.

    Applied by traversing the Flax parameter pytree and matching key names.
    """
    return {
        # Embeddings
        "embedding":        P(None, None),
        "pos_embedding":    P(None, None),
        # Attention projections — column-parallel on model axis
        "q_proj/kernel":    P(None, model_axis),
        "k_proj/kernel":    P(None, model_axis),
        "v_proj/kernel":    P(None, model_axis),
        # Output projection — row-parallel (transposed from column-parallel input)
        "out_proj/kernel":  P(model_axis, None),
        # MLP — column then row
        "Dense_0/kernel":   P(None, model_axis),
        "Dense_1/kernel":   P(model_axis, None),
        # LayerNorm — small, replicated
        "gamma":            P(None),
        "beta":             P(None),
        # LM head
        "lm_head/kernel":   P(None, None),
    }


# ─── Sharding Helpers ────────────────────────────────────────────────────────

def shard_params(params: Any, mesh: Mesh, data_axis: str = "data", model_axis: str = "model") -> Any:
    """
    Apply NamedSharding to a Flax parameter pytree.

    Walks the nested dict, matches leaf names against get_partition_specs(),
    and calls jax.device_put with the appropriate sharding.
    """
    specs = get_partition_specs(data_axis, model_axis)

    def _shard_leaf(path: tuple, leaf: Any) -> Any:
        key = "/".join(str(k) for k in path)
        for fragment, pspec in specs.items():
            if fragment in key:
                sharding = NamedSharding(mesh, pspec)
                return jax.device_put(leaf, sharding)
        # Default: replicate
        return jax.device_put(leaf, NamedSharding(mesh, P()))

    return jax.tree_util.tree_map_with_path(_shard_leaf, params)


def shard_batch(batch: Any, mesh: Mesh, data_axis: str = "data") -> Any:
    """Shard a data batch along the data_axis (first dimension)."""
    pspec    = P(data_axis, None)
    sharding = NamedSharding(mesh, pspec)
    return jax.device_put(batch, sharding)


def replicate(x: Any, mesh: Mesh) -> Any:
    """Replicate a pytree across all devices in the mesh."""
    sharding = NamedSharding(mesh, P())
    return jax.device_put(x, sharding)
