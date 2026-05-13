"""Tests for the observability stack: metrics store, profiler, gradient monitor."""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
import time


# ---------------------------------------------------------------------------
# MetricsStore
# ---------------------------------------------------------------------------

def test_metrics_store_push_and_retrieve():
    from observability.dashboard.metrics_store import MetricsStore

    store = MetricsStore(maxlen=100)
    store.push(step=1, loss=3.0, perplexity=20.0, grad_norm=1.2, lr=1e-4)
    store.push(step=2, loss=2.8, perplexity=18.0, grad_norm=0.9, lr=9e-5)

    data = store.get_recent(n=10)
    assert len(data["loss"]) == 2
    assert data["loss"][0] == pytest.approx(3.0)
    assert data["step"][-1] == 2


def test_metrics_store_maxlen():
    from observability.dashboard.metrics_store import MetricsStore

    store = MetricsStore(maxlen=5)
    for i in range(10):
        store.push(step=i, loss=float(i), perplexity=float(i), grad_norm=1.0, lr=1e-4)
    data = store.get_recent(n=100)
    assert len(data["step"]) == 5
    assert data["step"][0] == 5


def test_metrics_store_thread_safety():
    import threading
    from observability.dashboard.metrics_store import MetricsStore

    store = MetricsStore(maxlen=1000)
    errors = []

    def writer(start):
        try:
            for i in range(100):
                store.push(step=start + i, loss=float(i), perplexity=float(i), grad_norm=1.0, lr=1e-4)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i * 100,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors


# ---------------------------------------------------------------------------
# Kernel profiler
# ---------------------------------------------------------------------------

def test_time_kernel_returns_stats():
    from observability.kernel_profiler import time_kernel

    fn = jax.jit(lambda x: jnp.sum(x ** 2))
    x = jnp.ones((256,))
    stats = time_kernel(fn, x, n_trials=5)

    assert "median_ms" in stats
    assert "std_ms" in stats
    assert stats["median_ms"] > 0


def test_theoretical_flops_positive():
    from observability.kernel_profiler import theoretical_attention_flops

    flops = theoretical_attention_flops(batch=1, heads=8, seq=512, d_head=64)
    assert flops > 0


# ---------------------------------------------------------------------------
# Gradient monitor
# ---------------------------------------------------------------------------

def test_global_grad_norm():
    from observability.gradient_monitor import global_grad_norm

    grads = {"w": jnp.ones((4, 4)), "b": jnp.ones((4,))}
    norm = global_grad_norm(grads)
    expected = float(jnp.sqrt(4 * 4 + 4))
    assert abs(float(norm) - expected) < 1e-4


def test_compute_layer_grad_norms_keys():
    from observability.gradient_monitor import compute_layer_grad_norms

    grads = {
        "layer_0": {"w": jnp.ones((8, 8))},
        "layer_1": {"w": jnp.ones((8, 8))},
    }
    norms = compute_layer_grad_norms(grads)
    assert set(norms.keys()) == {"layer_0", "layer_1"}


# ---------------------------------------------------------------------------
# Memory tracker
# ---------------------------------------------------------------------------

def test_memory_tracker_context_manager():
    from observability.memory_tracker import MemoryTracker

    with MemoryTracker() as tracker:
        x = jnp.ones((1024, 1024))
        jax.block_until_ready(x)

    assert isinstance(tracker.peak_mb, float)
    assert tracker.peak_mb >= 0


def test_estimate_param_memory():
    from observability.memory_tracker import estimate_param_memory_mb

    params = {"w": jnp.ones((1024, 1024), jnp.float32)}
    mb = estimate_param_memory_mb(params)
    # 1024*1024*4 bytes = 4MB
    assert abs(mb - 4.0) < 0.1


# ---------------------------------------------------------------------------
# FLOP counter
# ---------------------------------------------------------------------------

def test_attention_flops_scaling():
    from observability.flops_counter import attention_flops

    f1 = attention_flops(batch=1, heads=8, seq=512, d_head=64)
    f2 = attention_flops(batch=1, heads=8, seq=1024, d_head=64)
    # Attention FLOPs scale as O(S^2), so doubling seq => 4x FLOPs
    assert abs(f2 / f1 - 4.0) < 0.1


def test_model_flops_per_step_positive():
    from observability.flops_counter import model_flops_per_step
    from model.config import ModelConfig

    cfg = ModelConfig(n_layers=2, n_heads=4, d_model=64, d_ff=128, vocab_size=256)
    flops = model_flops_per_step(cfg, seq_len=128, batch_size=1)
    assert flops > 0
