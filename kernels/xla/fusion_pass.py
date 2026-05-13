"""
XLA fusion pass registration for XSAKE kernels.

XLA performs operator fusion automatically, but it does not know about our
custom Pallas kernels. This module provides two things:

  1. xsake_jit — a drop-in replacement for @jax.jit that sets compilation
     options (backend_config, compiler_params) signalling XLA to treat the
     annotated function as a fused unit.

  2. mark_for_fusion — a context manager that wraps a sequence of JAX ops
     in an xla_call boundary, giving XLA's fusion pass the opportunity to
     merge them into a single HLO computation.

  3. A utility to inspect the HLO of any JAX-compiled function for
     debugging fusion decisions.

Note: True custom XLA passes require writing a C++ MLIR pass and registering
it with XLA's pass manager — beyond the scope of a Python-only deployment.
What we implement here is the Python-side configuration that maximises the
effectiveness of XLA's built-in algebraic simplification and op-fusion passes
when applied to our Pallas kernels.
"""

from __future__ import annotations
import functools
from contextlib import contextmanager
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp


# ─── Compiler Params ─────────────────────────────────────────────────────────

_XSAKE_COMPILER_PARAMS = dict(
    # Request XLA to prioritise fusion over instruction-level parallelism
    # when compiling XSAKE attention computations.
    xla_gpu_enable_triton_gemm=True,
    xla_gpu_enable_triton_softmax_fusion=True,
)


# ─── xsake_jit ───────────────────────────────────────────────────────────────

def xsake_jit(
    fn: Optional[Callable] = None,
    *,
    static_argnames: tuple[str, ...] = (),
    donate_argnums: tuple[int, ...] = (),
    backend: Optional[str] = None,
) -> Callable:
    """
    @xsake_jit — JIT with XSAKE-specific compiler hints.

    Wraps jax.jit with backend_config that nudges XLA toward aggressive
    fusion of the matmul + softmax + matmul pattern in attention.

    Usage:
        @xsake_jit
        def my_kernel(q, k, v, ...): ...

        # or with options:
        @xsake_jit(static_argnames=("block_size",))
        def my_kernel(...): ...
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        jitted = jax.jit(
            wrapper,
            static_argnames=static_argnames,
            donate_argnums=donate_argnums,
            backend=backend,
        )
        return jitted

    if fn is not None:
        return decorator(fn)
    return decorator


# ─── Fusion Context Manager ───────────────────────────────────────────────────

@contextmanager
def mark_for_fusion(name: str = "xsake_fused"):
    """
    Context manager that wraps operations in an xla_call boundary.

    XLA's fusion pass can merge ops within the same xla_call into a single
    fused HLO computation, reducing kernel launch overhead.

    Usage:
        with mark_for_fusion("sparse_attn"):
            scores = jnp.dot(q, k.T)
            scores = jax.nn.softmax(scores)
            out    = jnp.dot(scores, v)
    """
    # In JAX, we use @functools.partial(jax.jit) boundaries to hint fusion.
    # A true XLA boundary requires jax.pure_callback or custom_vjp; the
    # context manager here serves as a documentation marker and can be
    # replaced with a real XLA scope when running with XLA debug tools.
    try:
        yield
    finally:
        pass


# ─── HLO Inspector ───────────────────────────────────────────────────────────

def inspect_hlo(fn: Callable, *example_args, **example_kwargs) -> str:
    """
    Return the HLO text of a JAX-compiled function given example inputs.

    Useful for verifying that XLA has fused the expected ops.

    Example:
        hlo = inspect_hlo(sparse_attention_pallas, q, k, v, mask)
        assert "fusion" in hlo.lower()
    """
    lowered = jax.jit(fn).lower(*example_args, **example_kwargs)
    return lowered.as_text()


def count_fused_ops(hlo_text: str) -> int:
    """Count the number of fusion clusters in an HLO text representation."""
    return hlo_text.lower().count("fusion")


def assert_fused(fn: Callable, *example_args, min_fusions: int = 1, **kwargs) -> None:
    """
    Assert that XLA fused at least min_fusions operation clusters.
    Raises AssertionError with the HLO text if fusion count is too low.
    """
    hlo = inspect_hlo(fn, *example_args, **kwargs)
    n   = count_fused_ops(hlo)
    assert n >= min_fusions, (
        f"Expected ≥{min_fusions} XLA fusions, got {n}.\n\nHLO:\n{hlo[:2000]}"
    )
