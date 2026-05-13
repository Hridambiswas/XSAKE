"""
Device mesh construction for XSAKE distributed training.

XSAKE uses a 2D mesh: (data_parallel × model_parallel).

  data axis  — replicate model, shard data (standard DDP)
  model axis — tensor-parallel sharding of weight matrices

On a single device (Kaggle T4), both axes have size 1 and the
mesh is a no-op. On 8 TPU cores, a (4, 2) mesh means 4-way data
parallelism + 2-way model parallelism.
"""

from __future__ import annotations
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from model.config import MeshConfig


def make_mesh(config: MeshConfig) -> Mesh:
    """
    Create a 2D device mesh from MeshConfig.

    Devices are ordered row-major: all data-parallel replicas share
    the same model shard, enabling efficient allreduce within rows.

    Args:
        config: MeshConfig with n_devices, data_parallel, model_parallel

    Returns:
        jax.sharding.Mesh with axes (config.data_axis, config.model_axis)

    Raises:
        ValueError: if data_parallel * model_parallel != n_devices
    """
    dp = config.data_parallel
    mp = config.model_parallel
    n  = config.n_devices

    if dp * mp != n:
        raise ValueError(
            f"data_parallel ({dp}) × model_parallel ({mp}) = {dp * mp} "
            f"≠ n_devices ({n})"
        )

    devices = np.array(jax.devices()[:n]).reshape(dp, mp)
    return Mesh(devices, axis_names=(config.data_axis, config.model_axis))


def auto_mesh(data_axis: str = "data", model_axis: str = "model") -> Mesh:
    """
    Automatically create the largest valid 2D mesh from available devices.

    Prefers squarish meshes (data_parallel >= model_parallel).
    Falls back to (n_devices, 1) if no square factor is found.
    """
    n = jax.device_count()
    best_dp, best_mp = n, 1

    for mp in range(n, 0, -1):
        if n % mp == 0:
            dp = n // mp
            if dp >= mp:
                best_dp, best_mp = dp, mp
                break

    devices = np.array(jax.devices()).reshape(best_dp, best_mp)
    return Mesh(devices, axis_names=(data_axis, model_axis))


def mesh_info(mesh: Mesh) -> dict:
    """Return a summary dict of a mesh for logging."""
    return {
        "total_devices": mesh.size,
        "axes": {name: size for name, size in zip(mesh.axis_names, mesh.devices.shape)},
        "device_kind": jax.devices()[0].device_kind,
    }
