"""
JIT compilation utilities for XSAKE.

Provides helpers for compiling model functions with static argument handling,
donation of input buffers, and backend selection.

Also exposes a warm-up utility that triggers XLA compilation before the
actual training loop to avoid measuring compilation time in benchmarks.
"""

from __future__ import annotations
from functools import partial
from typing import Any, Callable, Optional, Sequence

import jax
import jax.numpy as jnp


def xsake_jit(
    fn: Callable,
    static_argnames: Sequence[str] = (),
    donate_argnums: Sequence[int] = (),
    backend: Optional[str] = None,
) -> Callable:
    """
    JIT-compile a function with XSAKE conventions.

    Args:
        fn:              function to compile
        static_argnames: argument names treated as compile-time constants
        donate_argnums:  argument indices whose buffers may be donated
        backend:         force a specific XLA backend ("gpu", "tpu", "cpu")

    Returns:
        JIT-compiled callable
    """
    return jax.jit(
        fn,
        static_argnames=tuple(static_argnames),
        donate_argnums=tuple(donate_argnums),
        backend=backend,
    )


def warmup(
    fn: Callable,
    *example_args,
    n_warmup: int = 3,
    **example_kwargs,
) -> None:
    """
    Run fn n_warmup times on example inputs to trigger XLA compilation.

    Call this before starting the timed benchmark loop.

    Args:
        fn:            JIT-compiled function
        *example_args: representative inputs (same shapes/dtypes as real data)
        n_warmup:      number of warmup iterations
    """
    for i in range(n_warmup):
        out = fn(*example_args, **example_kwargs)
        jax.block_until_ready(out)
    print(f"[jit] Warmup complete ({n_warmup} iterations, compilation done)")


def compile_time(
    fn: Callable,
    *example_args,
    **example_kwargs,
) -> float:
    """
    Measure XLA compilation time for fn on example inputs.

    Returns:
        compilation time in seconds
    """
    import time

    # Force compilation by running once
    t0 = time.perf_counter()
    out = fn(*example_args, **example_kwargs)
    jax.block_until_ready(out)
    return time.perf_counter() - t0


def lowered_hlo(fn: Callable, *example_args, **example_kwargs) -> str:
    """Return the HLO text representation of a JIT-compiled function."""
    return jax.jit(fn).lower(*example_args, **example_kwargs).as_text()


def device_memory_stats() -> dict:
    """Return per-device memory usage (GPU/TPU only)."""
    backend = jax.default_backend()
    if backend not in ("gpu", "tpu"):
        return {"backend": backend, "note": "memory stats only available on GPU/TPU"}

    stats = {}
    for i, dev in enumerate(jax.devices()):
        mem = dev.memory_stats() if hasattr(dev, "memory_stats") else {}
        stats[f"device_{i}"] = mem
    return stats
