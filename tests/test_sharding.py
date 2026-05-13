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
    assert "n_devices" in info
    assert "axes" in info


# ---------------------------------------------------------------------------
# Sharding: replicate + shard_batch
# ---------------------------------------------------------------------------

def test_replicate_preserves_values():
    from distributed.sharding import replicate

    arr = jnp.arange(16, dtype=jnp.float32)
    rep = replicate(arr)
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
    from model.config import ModelConfig

    cfg = ModelConfig(n_layers=2, n_heads=4, d_model=64, d_ff=128)
    specs = get_partition_specs(cfg)
    assert isinstance(specs, dict)


# ---------------------------------------------------------------------------
# pmap step smoke test (single device — pmap trivially works)
# ---------------------------------------------------------------------------

def test_pmap_step_output_shape():
    from distributed.pmap_trainer import make_pmapped_step, shard_batch_pmap
    from training.optimizer import make_optimizer, TrainState
    from model.transformer import XSAKETransformer
    from model.config import ModelConfig, TrainingConfig
    import optax

    model_cfg = ModelConfig(n_layers=1, n_heads=2, d_model=32, d_ff=64, vocab_size=256)
    train_cfg = TrainingConfig(batch_size=jax.device_count(), seq_len=16, max_steps=1)

    model = XSAKETransformer(model_cfg)
    key = jax.random.PRNGKey(0)
    dummy = jnp.ones((1, 16), dtype=jnp.int32)
    params = model.init(key, dummy)["params"]

    optimizer, schedule = make_optimizer(train_cfg)
    state = TrainState(
        step=jnp.array(0),
        params=params,
        opt_state=optimizer.init(params),
    )

    pmapped = make_pmapped_step(model, optimizer)
    n = jax.device_count()
    batch = shard_batch_pmap(
        {"input_ids": jnp.ones((n, 16), dtype=jnp.int32)}, n
    )
    keys = jax.random.split(key, n)

    loss, new_state = pmapped(state, batch, keys)
    assert loss.shape == (n,)
    assert new_state.step.shape == (n,)
