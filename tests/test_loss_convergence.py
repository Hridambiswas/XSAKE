"""
Training convergence smoke test.

Runs 20 gradient steps on synthetic data and asserts:
  1. Loss is finite at every step
  2. Final loss < initial loss (loss is decreasing)
  3. Final loss < 12.0 (model is learning, not stuck)

Runs on CPU in ~60 seconds. Used by the training_smoke_test.yml CI workflow.
"""

import json
import os
import pytest
import jax
import jax.numpy as jnp
import optax

from model.config import ModelConfig, TrainingConfig, KernelConfig, HADSConfig
from model.transformer import XSAKETransformer, cross_entropy_loss
from training.optimizer import make_optimizer, init_train_state
from training.mixed_precision import MixedPrecisionPolicy
from kernels.hads.hads_pattern import dummy_hads_profile


SMOKE_STEPS = int(os.environ.get("XSAKE_SMOKE_STEPS", "20"))


def _make_tiny_config():
    kernel = KernelConfig(
        block_size_q=32,
        block_size_kv=32,
        sparsity_type="local",
        use_pallas=False,
        hads=HADSConfig(block_size=32, window_size=1),
    )
    return ModelConfig(
        vocab_size=256,
        seq_len=64,
        n_layers=2,
        n_heads=4,
        d_model=64,
        d_ff=128,
        dropout=0.0,
        bias=False,
        dtype="float32",
        kernel=kernel,
    )


def _make_fake_batch(vocab_size=256, seq_len=64, batch=4):
    key = jax.random.PRNGKey(0)
    ids    = jax.random.randint(key, (batch, seq_len), 0, vocab_size)
    labels = jnp.roll(ids, -1, axis=-1).at[:, -1].set(-1)
    return {"input_ids": ids, "labels": labels}


def test_loss_decreases():
    model_cfg    = _make_tiny_config()
    training_cfg = TrainingConfig(
        batch_size=4, seq_len=64, max_steps=SMOKE_STEPS,
        warmup_steps=2, learning_rate=3e-3, weight_decay=0.0,
        grad_clip=1.0, precision="float32",
        gradient_checkpointing=False,
    )

    profile = dummy_hads_profile(
        n_heads=model_cfg.n_heads,
        seq_len=model_cfg.seq_len,
        block_size=model_cfg.kernel.block_size_q,
    )
    model     = XSAKETransformer(config=model_cfg, hads_profile=profile)
    optimizer = make_optimizer(training_cfg)
    policy    = MixedPrecisionPolicy(use_bf16=False)

    rng = jax.random.PRNGKey(42)
    dummy_ids = jnp.zeros((1, model_cfg.seq_len), dtype=jnp.int32)
    params = model.init({"params": rng, "dropout": rng}, dummy_ids)["params"]
    state  = init_train_state(params, optimizer, rng)

    batch = _make_fake_batch(model_cfg.vocab_size, model_cfg.seq_len)

    losses = []

    @jax.jit
    def step(state, batch):
        def loss_fn(p):
            logits, _ = model.apply({"params": p}, batch["input_ids"], training=False)
            return cross_entropy_loss(logits, batch["labels"]), logits

        (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        updates, new_opt = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        return state._replace(
            step=state.step + 1,
            params=new_params,
            opt_state=new_opt,
        ), loss

    for i in range(SMOKE_STEPS):
        state, loss = step(state, batch)
        loss_val = float(loss)
        assert jnp.isfinite(loss), f"Non-finite loss at step {i}: {loss_val}"
        losses.append(loss_val)

    initial_loss = losses[0]
    final_loss   = losses[-1]

    result = {"initial_loss": initial_loss, "final_loss": final_loss, "steps": SMOKE_STEPS}
    with open("smoke_test_result.json", "w") as f:
        json.dump(result, f)

    assert final_loss < initial_loss, \
        f"Loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"
    assert final_loss < 12.0, \
        f"Final loss too high: {final_loss:.4f} (threshold: 12.0)"

    print(f"\n✅ Loss: {initial_loss:.4f} → {final_loss:.4f} over {SMOKE_STEPS} steps")
