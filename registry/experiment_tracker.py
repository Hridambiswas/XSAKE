"""Local experiment tracker: logs hyperparameters and metrics to JSON."""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Experiment:
    exp_id: str
    name: str
    config: Dict[str, Any]
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    status: str = "running"
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    notes: str = ""

    def log(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            self.metrics.setdefault(k, []).append(float(v))

    def finish(self, status: str = "completed") -> None:
        self.status = status
        self.finished_at = time.time()

    def summary(self) -> Dict[str, float]:
        return {k: v[-1] for k, v in self.metrics.items() if v}


class ExperimentTracker:
    """JSON-backed experiment tracker."""

    def __init__(self, store_path: str = "registry/experiments.json"):
        self._path = Path(store_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._experiments: Dict[str, Experiment] = self._load()
        self._active: Optional[Experiment] = None

    def _load(self) -> Dict[str, Experiment]:
        if not self._path.exists():
            return {}
        with open(self._path) as f:
            raw = json.load(f)
        return {eid: Experiment(**e) for eid, e in raw.items()}

    def _save(self) -> None:
        with open(self._path, "w") as f:
            json.dump({eid: asdict(e) for eid, e in self._experiments.items()}, f, indent=2)

    def start(
        self,
        name: str,
        config: Dict[str, Any],
        tags: List[str] | None = None,
        notes: str = "",
    ) -> Experiment:
        exp_id = str(uuid.uuid4())[:8]
        exp = Experiment(exp_id=exp_id, name=name, config=config, tags=tags or [], notes=notes)
        self._experiments[exp_id] = exp
        self._active = exp
        self._save()
        return exp

    def log(self, **kwargs: float) -> None:
        if self._active is None:
            raise RuntimeError("No active experiment. Call start() first.")
        self._active.log(**kwargs)
        self._save()

    def finish(self, status: str = "completed") -> None:
        if self._active is None:
            return
        self._active.finish(status)
        self._active = None
        self._save()

    def get(self, exp_id: str) -> Optional[Experiment]:
        return self._experiments.get(exp_id)

    def list_experiments(self, status: Optional[str] = None) -> List[Experiment]:
        exps = list(self._experiments.values())
        if status:
            exps = [e for e in exps if e.status == status]
        return sorted(exps, key=lambda e: e.created_at, reverse=True)

    def best(self, metric: str, lower_is_better: bool = True) -> Optional[Experiment]:
        valid = [e for e in self._experiments.values() if metric in e.metrics and e.metrics[metric]]
        if not valid:
            return None
        key = lambda e: e.metrics[metric][-1]
        return min(valid, key=key) if lower_is_better else max(valid, key=key)

    def summary_table(self) -> str:
        rows = ["exp_id    name                   status     metrics"]
        rows.append("-" * 70)
        for exp in self.list_experiments():
            summary = "  ".join(f"{k}={v:.4f}" for k, v in exp.summary().items())
            rows.append(f"{exp.exp_id:9s} {exp.name:22s} {exp.status:10s} {summary}")
        return "\n".join(rows)
