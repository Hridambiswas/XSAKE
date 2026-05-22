"""Tests for distributed sharding utilities."""
import pytest
import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_params(rng_key, shapes: dict) -> dict:
    params = {}
    for name, shape in shapes.items():
        params[name] = jax.random.normal(rng_key, shape)
    return params


# ---------------------------------------------------------------------------
# Mesh tests (CPU: single device)
# ---------------------------------------------------------------------------

def test_make_mesh_single_device():
    from distributed.mesh import make_mesh, MeshConfig

    cfg = MeshConfig(n_devices=1, data_parallel=1, model_parallel=1)
    mesh = make_mesh(cfg)
    assert mesh.shape == {"data": 1, "model": 1}


def test_auto_mesh_returns_valid_shape():
    from distributed.mesh import auto_mesh

    mesh = auto_mesh()
    total = 1
    for v in mesh.shape.values():
        total *= v
    assert total == jax.device_count()


def test_mesh_info_keys():
    from distributed.mesh import auto_mesh, mesh_info

    mesh = auto_mesh()
    info = mesh_info(mesh)
    # mesh_info returns total_devices (not n_devices)
    assert "total_devices" in info
    assert "axes" in info


# ---------------------------------------------------------------------------
# Sharding: replicate + shard_batch
# ---------------------------------------------------------------------------

def test_replicate_preserves_values():
    from distributed.sharding import replicate
    from distributed.mesh import auto_mesh

    mesh = auto_mesh()
    arr = jnp.arange(16, dtype=jnp.float32)
    rep = replicate(arr, mesh)
    np.testing.assert_array_equal(np.array(rep), np.array(arr))


def test_shard_batch_shape():
    from distributed.sharding import shard_batch
    from distributed.mesh import auto_mesh

    mesh = auto_mesh()
    n_dev = jax.device_count()
    batch = {"input_ids": jnp.ones((n_dev * 4, 16), dtype=jnp.int32)}
    sharded = shard_batch(batch, mesh)
    # Shape should remain the same; sharding is a logical property
    assert sharded["input_ids"].shape == (n_dev * 4, 16)


# ---------------------------------------------------------------------------
# Partition specs
# ---------------------------------------------------------------------------

def test_partition_specs_keys():
    from distributed.sharding import get_partition_specs

    specs = get_partition_specs()
    assert isinstance(specs, dict)


# ---------------------------------------------------------------------------
# pmap step smoke test (single device — pmap trivially works)
# ---------------------------------------------------------------------------

def test_pmap_step_output_shape():
    from distributed.pmap_trainer import make_pmapped_step, shard_batch_pmap
    from flax.jax_utils import replicate as flax_replicate
    from training.optimizer import make_optimizer, init_train_state
    from training.mixed_precision import MixedPrecisionPolicy
    from model.transformer import XSAKETransformer
    from model.config import ModelConfig, TrainingConfig, KernelConfig

    # block_size must be <= seq_len; use dense sparsity on CPU (no Pallas)
    kernel_cfg = KernelConfig(
        block_size_q=16, block_size_kv=16,
        use_pallas=False, sparsity_type="dense",
    )
    model_cfg = ModelConfig(
        n_layers=1, n_heads=2, d_model=32, d_ff=64,
        vocab_size=256, kernel=kernel_cfg,
    )
    train_cfg = TrainingConfig(batch_size=jax.device_count(), seq_len=16, max_steps=1)

    model = XSAKETransformer(model_cfg)
    key = jax.random.PRNGKey(0)
    dummy = jnp.ones((1, 16), dtype=jnp.int32)
    params = model.init(key, dummy)["params"]

    optimizer = make_optimizer(train_cfg)
    state = init_train_state(params, optimizer, jax.random.PRNGKey(1))

    policy = MixedPrecisionPolicy()
    pmapped = make_pmapped_step(model, optimizer, policy)

    n = jax.device_count()
    batch = shard_batch_pmap({
        "input_ids": jnp.ones((n, 16), dtype=jnp.int32),
        "labels":    jnp.ones((n, 16), dtype=jnp.int32),
    })

    rep_state = flax_replicate(state)
    new_state, metrics = pmapped(rep_state, batch)
    assert metrics["loss"].shape == (n,)
    assert new_state.step.shape == (n,)
