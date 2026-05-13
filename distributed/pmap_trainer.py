"""
pmap-based multi-device training loop for XSAKE.

pmap replicates the training step across all local devices and performs
a synchronised allreduce on gradients. It is the lower-level alternative
to the NamedSharding approach in trainer.py, more appropriate when running
on a single multi-GPU node or TPU pod where all devices are local.

This module provides:
  pmapped_train_step — single step, gradient-synced across devices
  replicate_state    — distribute TrainState to all devices
  unreplicate_state  — gather TrainState back to a single device

Use trainer.py for single-device or NamedSharding runs.
Use this module when jax.device_count() > 1 and all devices are local.
"""

from __future__ import annotations
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax.jax_utils import replicate, unreplicate

from model.transformer import XSAKETransformer, cross_entropy_loss
from training.optimizer import TrainState
from training.mixed_precision import MixedPrecisionPolicy


def _pmap_step(
    state: TrainState,
    batch: dict,
    model: XSAKETransformer,
    optimizer: optax.GradientTransformation,
    policy: MixedPrecisionPolicy,
) -> tuple[TrainState, dict]:
    """Inner step — runs identically on each device."""

    def loss_fn(params):
        params_bf16 = policy.cast_params(params)
        logits, _   = model.apply(
            {"params": params_bf16},
            batch["input_ids"],
            training=True,
            rngs={"dropout": state.rng},
        )
        return cross_entropy_loss(logits, batch["labels"]), logits

    (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Synchronise gradients across devices via allreduce mean
    grads = jax.lax.pmean(grads, axis_name="devices")
    loss  = jax.lax.pmean(loss,  axis_name="devices")
    grads = policy.cast_grads(grads)

    updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    new_rng, _ = jax.random.split(state.rng)
    new_state  = TrainState(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        rng=new_rng,
    )
    return new_state, {"loss": loss, "grad_norm": optax.global_norm(grads)}


def make_pmapped_step(
    model: XSAKETransformer,
    optimizer: optax.GradientTransformation,
    policy: MixedPrecisionPolicy,
):
    """
    Return a pmap-compiled training step.

    The returned function expects TrainState and batch to be already
    replicated across devices (use replicate_state() below).
    """
    return jax.pmap(
        partial(_pmap_step, model=model, optimizer=optimizer, policy=policy),
        axis_name="devices",
        donate_argnums=(0,),   # donate state buffer for in-place update
    )


def replicate_state(state: TrainState) -> TrainState:
    """Replicate a single-device TrainState across all local devices."""
    return replicate(state)


def unreplicate_state(state: TrainState) -> TrainState:
    """Gather the replicated TrainState back to a single device."""
    return unreplicate(state)


def shard_batch_pmap(batch: dict) -> dict:
    """
    Reshape a batch for pmap: add a leading device axis.

    [batch, seq] → [n_devices, batch//n_devices, seq]
    """
    n_devices = jax.local_device_count()
    return jax.tree_util.tree_map(
        lambda x: x.reshape(n_devices, x.shape[0] // n_devices, *x.shape[1:]),
        batch,
    )
