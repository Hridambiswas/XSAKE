"""
XLA compiler flags and environment configuration for XSAKE.

These flags are set before JAX initialises to tune XLA's compilation
behaviour for sparse attention workloads on GPU and TPU.

Key levers:
  - XLA_FLAGS: controls auto-tuning, fusion aggressiveness, memory limits
  - JAX_ENABLE_X64: disabled (bfloat16 training policy)
  - TF_CPP_MIN_LOG_LEVEL: suppress TensorFlow init noise
  - NCCL flags: multi-GPU communication tuning

Call setup_xla_flags() at the top of any training or benchmark script
before importing jax.
"""

from __future__ import annotations
import os
from typing import Literal

Backend = Literal["gpu", "tpu", "cpu"]


# ─── Flag Bundles ─────────────────────────────────────────────────────────────

_GPU_FLAGS = [
    "--xla_gpu_enable_triton_softmax_fusion=true",
    "--xla_gpu_enable_triton_gemm=true",
    "--xla_gpu_enable_async_collectives=true",
    "--xla_gpu_enable_latency_hiding_scheduler=true",
    "--xla_gpu_enable_highest_priority_async_stream=true",
    "--xla_gpu_all_reduce_combine_threshold_bytes=134217728",
    "--xla_gpu_graph_level=0",             # disable CUDA graph to simplify profiling
    "--xla_force_host_platform_device_count=1",
]

_TPU_FLAGS = [
    "--xla_tpu_enable_async_collective_fusion=true",
    "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true",
    "--xla_tpu_megacore_fusion_allow_ags=false",
    "--xla_enable_async_all_gather=true",
    "--xla_enable_async_collective_permute=true",
]

_CPU_FLAGS: list[str] = []   # no special flags needed for CPU


# ─── Setup ───────────────────────────────────────────────────────────────────

def setup_xla_flags(backend: Backend = "gpu", debug: bool = False) -> None:
    """
    Configure XLA environment variables.

    Must be called BEFORE `import jax` or any JAX operation.

    Args:
        backend: target hardware ("gpu", "tpu", "cpu")
        debug:   if True, enables XLA_DUMP_TO for HLO inspection
    """
    flags: list[str]
    if backend == "gpu":
        flags = _GPU_FLAGS.copy()
    elif backend == "tpu":
        flags = _TPU_FLAGS.copy()
    else:
        flags = _CPU_FLAGS.copy()

    existing = os.environ.get("XLA_FLAGS", "")
    combined = " ".join([existing] + flags).strip()
    os.environ["XLA_FLAGS"] = combined

    # Suppress TF init noise
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    # Disable 64-bit since we train in bfloat16
    os.environ["JAX_ENABLE_X64"] = "0"

    # NCCL tuning for multi-GPU
    if backend == "gpu":
        os.environ.setdefault("NCCL_LAUNCH_MODE", "PARALLEL")
        os.environ.setdefault("NCCL_ALGO", "Tree")

    if debug:
        dump_dir = "/tmp/xsake_hlo"
        os.makedirs(dump_dir, exist_ok=True)
        os.environ["XLA_FLAGS"] += f" --xla_dump_to={dump_dir}"
        os.environ["XLA_FLAGS"] += " --xla_dump_hlo_as_text"
        print(f"[XLA] HLO dumps enabled → {dump_dir}")


def get_active_flags() -> str:
    """Return the current XLA_FLAGS string."""
    return os.environ.get("XLA_FLAGS", "(none)")


def print_config(backend: Backend = "gpu") -> None:
    """Print the full XLA configuration that would be applied."""
    flags = {"gpu": _GPU_FLAGS, "tpu": _TPU_FLAGS, "cpu": _CPU_FLAGS}[backend]
    print(f"XLA flags for {backend}:")
    for f in flags:
        print(f"  {f}")
