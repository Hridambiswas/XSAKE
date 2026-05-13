"""
Theoretical vs actual FLOPs measurement for XSAKE.

Theoretical FLOPs:
  - Attention: 2 * B * H * S * S * D_head * (1 - sparsity) for QK + AV
  - MLP:       2 * B * S * D_model * D_ff  (two matmuls, factor 2 for mul+add)
  - Full model: sum over all layers

Actual FLOPs:
  Measured via XLA's HLO computation count (requires debug compilation).
  On GPU, approximated from TFLOPS and wall time.

MFU (Model FLOP Utilisation):
  actual_TFLOPS / device_peak_TFLOPS
  T4 bfloat16 peak: 65 TFLOPS (Tensor Cores)
"""

from __future__ import annotations
from model.config import ModelConfig
from kernels.hads.hads_pattern import HADSProfile


def attention_flops(
    batch: int,
    heads: int,
    seq: int,
    d_head: int,
    mean_sparsity: float = 0.0,
) -> int:
    """FLOPs for one attention layer forward pass."""
    active = 1.0 - mean_sparsity
    qk = 2 * batch * heads * seq * seq * d_head * active
    av = 2 * batch * heads * seq * seq * d_head * active
    proj = 3 * 2 * batch * seq * (heads * d_head) * (heads * d_head)  # Q, K, V projections
    out  = 2 * batch * seq * (heads * d_head) * (heads * d_head)      # output projection
    return int(qk + av + proj + out)


def mlp_flops(batch: int, seq: int, d_model: int, d_ff: int) -> int:
    """FLOPs for one MLP layer (two dense matmuls)."""
    return int(2 * 2 * batch * seq * d_model * d_ff)


def model_flops_per_step(
    config: ModelConfig,
    batch: int,
    hads_profile: HADSProfile | None = None,
) -> dict:
    """
    Estimate total FLOPs for one forward pass through the full model.

    Args:
        config:       ModelConfig
        batch:        batch size
        hads_profile: if provided, uses per-head mean sparsity for FLOP estimate

    Returns:
        dict with attn_flops, mlp_flops, total_flops, tflops
    """
    seq    = config.seq_len
    H      = config.n_heads
    D      = config.d_head
    D_m    = config.d_model
    D_ff   = config.d_ff
    L      = config.n_layers

    mean_sparsity = 0.0
    if hads_profile is not None:
        mean_sparsity = float(hads_profile.head_sparsity.mean())

    attn_total = L * attention_flops(batch, H, seq, D, mean_sparsity)
    mlp_total  = L * mlp_flops(batch, seq, D_m, D_ff)
    total      = attn_total + mlp_total

    return {
        "batch":          batch,
        "seq_len":        seq,
        "n_layers":       L,
        "mean_sparsity":  round(mean_sparsity, 4),
        "attn_flops":     attn_total,
        "mlp_flops":      mlp_total,
        "total_flops":    total,
        "tflops":         round(total / 1e12, 4),
    }


def mfu(
    total_flops: int,
    step_time_s: float,
    device_peak_tflops: float = 65.0,
) -> float:
    """
    Model FLOP Utilisation.

    Args:
        total_flops:        from model_flops_per_step()["total_flops"]
        step_time_s:        measured wall time per step in seconds
        device_peak_tflops: device theoretical peak (T4 = 65, A100 = 312)

    Returns:
        MFU as a fraction (0–1)
    """
    achieved_tflops = total_flops / step_time_s / 1e12
    return achieved_tflops / device_peak_tflops
