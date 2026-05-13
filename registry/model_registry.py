"""Model artifact registry: track versions, configs, and checkpoint paths."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ModelRecord:
    model_id: str
    version: int
    checkpoint_path: str
    config: Dict
    metrics: Dict
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    notes: str = ""


class ModelRegistry:
    """JSON-backed model registry stored at `registry_path`."""

    def __init__(self, registry_path: str = "registry/model_registry.json"):
        self._path = Path(registry_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, List[ModelRecord]] = self._load()

    def _load(self) -> Dict[str, List[ModelRecord]]:
        if not self._path.exists():
            return {}
        with open(self._path) as f:
            raw = json.load(f)
        return {
            mid: [ModelRecord(**r) for r in versions]
            for mid, versions in raw.items()
        }

    def _save(self) -> None:
        with open(self._path, "w") as f:
            json.dump(
                {mid: [asdict(r) for r in versions] for mid, versions in self._records.items()},
                f,
                indent=2,
            )

    def register(
        self,
        model_id: str,
        checkpoint_path: str,
        config: Dict,
        metrics: Dict,
        tags: List[str] | None = None,
        notes: str = "",
    ) -> ModelRecord:
        versions = self._records.setdefault(model_id, [])
        version = len(versions) + 1
        record = ModelRecord(
            model_id=model_id,
            version=version,
            checkpoint_path=checkpoint_path,
            config=config,
            metrics=metrics,
            tags=tags or [],
            notes=notes,
        )
        versions.append(record)
        self._save()
        return record

    def latest(self, model_id: str) -> Optional[ModelRecord]:
        versions = self._records.get(model_id, [])
        return versions[-1] if versions else None

    def list_models(self) -> List[str]:
        return list(self._records.keys())

    def get_version(self, model_id: str, version: int) -> Optional[ModelRecord]:
        for r in self._records.get(model_id, []):
            if r.version == version:
                return r
        return None

    def best_by_metric(self, model_id: str, metric: str, lower_is_better: bool = True) -> Optional[ModelRecord]:
        records = self._records.get(model_id, [])
        valid = [r for r in records if metric in r.metrics]
        if not valid:
            return None
        return min(valid, key=lambda r: r.metrics[metric]) if lower_is_better else max(valid, key=lambda r: r.metrics[metric])

    def summary(self) -> str:
        lines = [f"Model Registry — {len(self._records)} model(s)"]
        for mid, versions in self._records.items():
            latest = versions[-1]
            lines.append(f"  {mid}: {len(versions)} version(s), latest v{latest.version}")
        return "\n".join(lines)
