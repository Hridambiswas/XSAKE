"""
Orbax-based model checkpointing for XSAKE.

Saves and restores:
  - Model parameters (fp32 master weights)
  - Optimizer state
  - Training step and RNG
  - Model and training config
  - HADS profile (if attached)

Versioned by step number. Keeps the N most recent checkpoints (default 3).
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from training.optimizer import TrainState
from model.config import ModelConfig, TrainingConfig


# ─── Checkpoint Manager ───────────────────────────────────────────────────────

def make_checkpoint_manager(
    directory: str,
    max_to_keep: int = 3,
) -> ocp.CheckpointManager:
    """
    Create an Orbax CheckpointManager that saves the last max_to_keep checkpoints.
    """
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        save_interval_steps=1,   # caller controls frequency
    )
    return ocp.CheckpointManager(
        directory=os.path.abspath(directory),
        checkpointers=ocp.PyTreeCheckpointer(),
        options=options,
    )


# ─── Save / Restore ──────────────────────────────────────────────────────────

def save_checkpoint(
    manager: ocp.CheckpointManager,
    state: TrainState,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> None:
    """
    Save a training checkpoint.

    Args:
        manager:         Orbax CheckpointManager
        state:           current TrainState (params, opt_state, step, rng)
        model_config:    for reconstruction on load
        training_config: for reconstruction on load
    """
    step = int(state.step)

    # Orbax serialises JAX pytrees natively
    ckpt = {
        "params":    state.params,
        "opt_state": state.opt_state,
        "step":      state.step,
        "rng":       state.rng,
    }
    manager.save(step, ckpt)

    # Save configs alongside the checkpoint as JSON
    ckpt_dir = Path(manager.directory) / str(step)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "model_config.json", "w") as f:
        json.dump(model_config.__dict__, f, indent=2, default=str)
    with open(ckpt_dir / "training_config.json", "w") as f:
        json.dump(training_config.__dict__, f, indent=2, default=str)


def restore_checkpoint(
    manager: ocp.CheckpointManager,
    state_template: TrainState,
    step: Optional[int] = None,
) -> TrainState:
    """
    Restore a checkpoint.

    Args:
        manager:        Orbax CheckpointManager
        state_template: TrainState with correct pytree structure (shapes/dtypes)
        step:           specific step to restore; if None, restores latest

    Returns:
        Restored TrainState
    """
    target_step = step if step is not None else manager.latest_step()
    if target_step is None:
        raise FileNotFoundError(f"No checkpoints found in {manager.directory}")

    template = {
        "params":    state_template.params,
        "opt_state": state_template.opt_state,
        "step":      state_template.step,
        "rng":       state_template.rng,
    }
    restored = manager.restore(target_step, items=template)

    return TrainState(
        step=restored["step"],
        params=restored["params"],
        opt_state=restored["opt_state"],
        rng=restored["rng"],
    )


def latest_step(directory: str) -> Optional[int]:
    """Return the latest checkpoint step in a directory, or None."""
    manager = make_checkpoint_manager(directory, max_to_keep=1)
    return manager.latest_step()
