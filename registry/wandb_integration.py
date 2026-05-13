"""Weights & Biases integration for XSAKE training runs."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional


_wandb_available = False
try:
    import wandb
    _wandb_available = True
except ImportError:
    pass


class WandbLogger:
    """Thin wrapper around wandb.init/log with a no-op fallback when W&B is unavailable."""

    def __init__(
        self,
        project: str = "xsake",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        enabled: bool = True,
    ):
        self._enabled = enabled and _wandb_available
        self._run = None

        if self._enabled:
            api_key = os.environ.get("WANDB_API_KEY")
            if api_key:
                wandb.login(key=api_key, relogin=False)

            self._run = wandb.init(
                project=project,
                entity=entity,
                name=name,
                config=config or {},
                tags=tags or [],
                reinit=True,
            )
            print(f"[W&B] Run URL: {self._run.get_url()}")
        else:
            print("[W&B] Logging disabled (wandb not installed or enabled=False).")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._enabled and self._run is not None:
            wandb.log(metrics, step=step)

    def log_artifact(self, path: str, name: str, artifact_type: str = "model") -> None:
        if not (self._enabled and self._run is not None):
            return
        artifact = wandb.Artifact(name=name, type=artifact_type)
        artifact.add_file(path)
        self._run.log_artifact(artifact)

    def watch(self, model_params: Dict, log_freq: int = 100) -> None:
        """Log parameter histograms (best-effort, skipped if not on GPU)."""
        if not (self._enabled and self._run is not None):
            return
        # wandb.watch requires PyTorch modules — log param norms manually instead
        import jax.numpy as jnp
        flat = {k: float(jnp.linalg.norm(v)) for k, v in model_params.items() if hasattr(v, "shape")}
        wandb.log({"param_norms/" + k: v for k, v in flat.items()})

    def finish(self) -> None:
        if self._enabled and self._run is not None:
            self._run.finish()
            self._run = None

    @property
    def run_id(self) -> Optional[str]:
        if self._run is not None:
            return self._run.id
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.finish()


def make_wandb_logger(
    config: Dict[str, Any],
    run_name: Optional[str] = None,
    tags: Optional[list] = None,
) -> WandbLogger:
    return WandbLogger(
        project=os.environ.get("WANDB_PROJECT", "xsake"),
        entity=os.environ.get("WANDB_ENTITY"),
        name=run_name,
        config=config,
        tags=tags,
        enabled=_wandb_available and bool(os.environ.get("WANDB_API_KEY")),
    )
