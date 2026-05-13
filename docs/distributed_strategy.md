# Distributed Training Strategy

## Hardware Assumptions

| Context | Hardware | Backend |
|---------|----------|---------|
| Local dev | Mac M4 Air | JAX CPU (1 device) |
| Kaggle GPU | NVIDIA T4 (2×) | JAX CUDA |
| Production target | TPU v4-8 / v5e | JAX TPU |

All code runs correctly on any of these. Backend is detected at runtime via `jax.devices()[0].platform`.

## Parallelism Strategy

XSAKE supports two parallelism axes organized as a 2D device mesh:

```
MeshConfig(n_devices=N, data_parallel=D, model_parallel=M)
# where D × M = N
```

### Data Parallelism (primary)

- Batch is split across the `"data"` mesh axis
- Each device holds a full replica of model parameters
- Gradients are synchronized via `jax.lax.pmean` after backward pass
- Implemented in `distributed/pmap_trainer.py`

### Model Parallelism (secondary, off by default)

- Large weight matrices sharded across the `"model"` axis
- `PartitionSpec("model", None)` for column-parallel; `PartitionSpec(None, "model")` for row-parallel
- Only activated when `MeshConfig.model_parallel > 1`
- Implemented in `distributed/sharding.py`

## Sharding Annotations

```python
# Replicated across all devices (embeddings, layernorm)
PartitionSpec(None, None)

# Data-parallel (batch dim on "data" axis)
PartitionSpec("data", None)

# Column-parallel weight (sharded on "model" axis)
PartitionSpec(None, "model")
```

`distributed/sharding.py:get_partition_specs()` maps ModelConfig to a flat dict of PartitionSpecs for every parameter.

## pmap vs jit+shard

- **Single GPU / CPU**: use `jax.jit` directly (no mesh needed)
- **Multi-GPU (≤8)**: use `jax.pmap` via `make_pmapped_step()`
- **TPU pod (>8 devices)**: use `jax.jit` + `NamedSharding` via `distributed/jit_compiler.py`

The trainer selects the strategy based on `jax.device_count()` at startup.

## Gradient Accumulation

Not yet implemented. Planned for seq_len > 4096 on single-GPU setups. Gradient accumulation steps would be added as an outer loop in `training/trainer.py:_train_step()`.

## Checkpoint Sharding

`training/checkpointing.py` uses Orbax with `StandardCheckpointer`. On multi-device setups, each device saves its shard independently; `restore_checkpoint()` reassembles via `mesh_restore`. This avoids the "gather all to host" bottleneck.

## Known Limitations

1. Model parallelism is untested on actual multi-GPU hardware (F-003 in `failure_analysis/FAILURES.md`).
2. `pmap` requires `batch_size % n_devices == 0`; the data loader does not yet enforce this automatically.
3. FSDP (fully sharded data parallelism) is not implemented — would require `jax.experimental.mesh_utils.create_hybrid_device_mesh`.
