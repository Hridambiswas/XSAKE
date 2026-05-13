"""
In-memory metrics store for the XSAKE dashboard.

Stores training metrics as a ring buffer (last N steps) and exposes
them to the FastAPI dashboard endpoints. Also publishes to Prometheus
for Grafana integration.

Thread-safe for concurrent reads from the dashboard and writes from
the training loop.
"""

from __future__ import annotations
import threading
from collections import deque
from typing import Any

from prometheus_client import Gauge, Counter, Histogram, CollectorRegistry


# ─── Prometheus registry ──────────────────────────────────────────────────────

registry = CollectorRegistry()

TRAIN_LOSS      = Gauge("xsake_train_loss",      "Training cross-entropy loss",   registry=registry)
TRAIN_PPL       = Gauge("xsake_train_ppl",        "Training perplexity",           registry=registry)
GRAD_NORM       = Gauge("xsake_grad_norm",         "Gradient L2 norm",              registry=registry)
TOKENS_SEEN     = Counter("xsake_tokens_total",    "Total tokens processed",        registry=registry)
STEP_LATENCY    = Histogram("xsake_step_latency_ms", "Per-step wall time (ms)",
                            buckets=[50, 100, 200, 500, 1000, 2000, 5000], registry=registry)
KERNEL_LATENCY  = Gauge("xsake_kernel_latency_ms", "Sparse attention kernel latency", registry=registry)
MEMORY_MB       = Gauge("xsake_memory_mb",         "Peak HBM usage (MB)",           registry=registry)
ACTIVE_BLOCKS   = Gauge("xsake_active_blocks_frac","HADS active block fraction",    registry=registry)


# ─── Ring-buffer store ────────────────────────────────────────────────────────

class MetricsStore:
    """Thread-safe ring buffer of training metrics."""

    def __init__(self, maxlen: int = 2000):
        self._lock   = threading.Lock()
        self._steps:  deque = deque(maxlen=maxlen)
        self._losses: deque = deque(maxlen=maxlen)
        self._ppls:   deque = deque(maxlen=maxlen)
        self._gnorms: deque = deque(maxlen=maxlen)
        self._tokens: deque = deque(maxlen=maxlen)

    def push(self, metrics: dict) -> None:
        """
        Record one step of training metrics.

        Expected keys: step, loss, perplexity, grad_norm, tokens
        Also updates Prometheus gauges.
        """
        with self._lock:
            self._steps.append(metrics.get("step", 0))
            loss = metrics.get("loss", float("nan"))
            ppl  = metrics.get("perplexity", float("nan"))
            gnorm = metrics.get("grad_norm", 0.0)
            tok  = metrics.get("tokens", 0)

            self._losses.append(loss)
            self._ppls.append(ppl)
            self._gnorms.append(gnorm)
            self._tokens.append(tok)

        # Update Prometheus (no lock needed — prometheus_client is thread-safe)
        if loss == loss:   # not NaN
            TRAIN_LOSS.set(loss)
            TRAIN_PPL.set(ppl)
            GRAD_NORM.set(gnorm)
        if metrics.get("step_ms"):
            STEP_LATENCY.observe(metrics["step_ms"])
        if metrics.get("kernel_latency_ms"):
            KERNEL_LATENCY.set(metrics["kernel_latency_ms"])
        if metrics.get("memory_mb"):
            MEMORY_MB.set(metrics["memory_mb"])
        if metrics.get("active_blocks"):
            ACTIVE_BLOCKS.set(metrics["active_blocks"])

    def get_metrics(self) -> dict:
        """Return all stored metrics as lists for JSON serialisation."""
        with self._lock:
            return {
                "steps":     list(self._steps),
                "losses":    list(self._losses),
                "ppls":      list(self._ppls),
                "grad_norms": list(self._gnorms),
                "tokens":    list(self._tokens),
            }

    def latest(self) -> dict:
        """Return the most recent step's metrics."""
        with self._lock:
            if not self._steps:
                return {}
            return {
                "step":      self._steps[-1],
                "loss":      self._losses[-1],
                "ppl":       self._ppls[-1],
                "grad_norm": self._gnorms[-1],
                "tokens":    self._tokens[-1],
            }
