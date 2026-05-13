"""
XLA / HLO kernel execution trace capture for XSAKE.

Provides utilities to:
  - Capture XLA profiling traces using JAX's built-in profiler
  - Extract per-op timing from HLO execution profiles
  - Measure kernel launch overhead vs compute time
  - Report FLOP utilisation (theoretical vs actual)

Usage on Kaggle T4:
    from observability.kernel_profiler import profile_attention

    with profile_attention(log_dir="/tmp/xsake_profile"):
        out = sparse_attention_pallas(q, k, v, mask)

Then open Tensorboard at /tmp/xsake_profile to inspect XLA traces.
"""

from __future__ import annotations
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

import jax
import jax.numpy as jnp


# ─── Trace context manager ────────────────────────────────────────────────────

@contextmanager
def xla_trace(log_dir: str = "/tmp/xsake_profile", duration_ms: int = 2000):
    """
    Capture an XLA execution trace to log_dir.

    Open with: tensorboard --logdir /tmp/xsake_profile
    Then visit: http://localhost:6006/#profile
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    jax.profiler.start_trace(log_dir)
    try:
        yield
    finally:
        jax.profiler.stop_trace()
        print(f"[profiler] Trace saved to {log_dir}")


# ─── Timing utilities ─────────────────────────────────────────────────────────

def time_kernel(
    fn: Callable,
    *args,
    n_warmup: int = 10,
    n_trials: int = 50,
    sync: bool = True,
) -> dict:
    """
    Precise kernel timing with device synchronisation.

    Args:
        fn:       compiled JAX function
        *args:    inputs
        n_warmup: warmup iterations (compilation + cache warm)
        n_trials: timed iterations
        sync:     block_until_ready between iterations

    Returns:
        dict with mean_ms, median_ms, std_ms, min_ms, max_ms
    """
    import numpy as np

    for _ in range(n_warmup):
        out = fn(*args)
        if sync:
            jax.block_until_ready(out)

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        out = fn(*args)
        if sync:
            jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return {
        "mean_ms":   float(times.mean()),
        "median_ms": float(np.median(times)),
        "std_ms":    float(times.std()),
        "min_ms":    float(times.min()),
        "max_ms":    float(times.max()),
        "n_trials":  n_trials,
    }


# ─── FLOP counting ────────────────────────────────────────────────────────────

def theoretical_attention_flops(
    batch: int,
    heads: int,
    seq: int,
    d_head: int,
    sparsity: float = 0.0,
) -> int:
    """
    Theoretical FLOPs for multi-head attention.

    Dense:  2 * B * H * S * S * D  (QK^T)  +  2 * B * H * S * S * D  (AV)
    Sparse: multiply by (1 - sparsity)

    Args:
        sparsity: fraction of (Q, KV) block pairs skipped

    Returns:
        total FLOPs (int)
    """
    qk_flops = 2 * batch * heads * seq * seq * d_head
    av_flops  = 2 * batch * heads * seq * seq * d_head
    total = (qk_flops + av_flops) * (1 - sparsity)
    return int(total)


def measure_arithmetic_utilisation(
    fn: Callable,
    *args,
    theoretical_flops: int,
    n_warmup: int = 5,
    n_trials: int = 20,
    device_tflops: float = 8.0,   # T4 = 8 TFLOPS bfloat16
) -> dict:
    """
    Compute arithmetic utilisation: actual TFLOPS / device peak TFLOPS.

    Args:
        theoretical_flops: computed by theoretical_attention_flops()
        device_tflops:     device peak TFLOPS (T4 bfloat16 ≈ 8, A100 ≈ 312)

    Returns:
        dict with achieved_tflops, utilisation_pct
    """
    timing = time_kernel(fn, *args, n_warmup=n_warmup, n_trials=n_trials)
    elapsed_s = timing["median_ms"] / 1000
    achieved  = theoretical_flops / elapsed_s / 1e12  # TFLOPS
    util_pct  = achieved / device_tflops * 100

    return {
        **timing,
        "theoretical_flops": theoretical_flops,
        "achieved_tflops":   round(achieved, 3),
        "peak_tflops":       device_tflops,
        "utilisation_pct":   round(util_pct, 1),
    }
