"""Small experiment-recording helpers for the public synthetic workflow.

The recorder always writes local JSON/CSV/Markdown artifacts. If MLflow is installed,
callers can additionally mirror params and metrics to a local MLflow run without making
MLflow mandatory for the lightweight public workflow.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


class ExperimentRecorder:
    """Create reproducible local experiment records under an artifact directory."""

    def __init__(self, output_dir: str | os.PathLike[str] = "artifacts/experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        path = self.output_dir / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_record(
        self,
        run_id: str,
        params: Mapping[str, Any],
        metrics: Mapping[str, Any],
        artifacts: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        path = self.run_dir(run_id)
        metadata = {
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "params": dict(params),
            "artifacts": dict(artifacts or {}),
        }
        write_json(path / "metadata.json", metadata)
        write_json(path / "metrics.json", dict(metrics))
        return path


def try_log_mlflow(
    experiment_name: str,
    run_name: str,
    params: Mapping[str, Any],
    metrics: Mapping[str, Any],
    artifact_paths: Iterable[str | os.PathLike[str]],
) -> bool:
    """Mirror a local record to MLflow when MLflow is available.

    Returns True if MLflow logging happened, otherwise False. The public workflow stays
    usable without MLflow so contributors can run the synthetic workflow after installing
    only the base simulator dependencies.
    """

    try:
        import mlflow  # type: ignore
    except Exception:
        return False

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: str(v) for k, v in params.items()})
        numeric_metrics = {}
        for key, value in metrics.items():
            try:
                numeric_metrics[key] = float(value)
            except (TypeError, ValueError):
                continue
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics)
        for artifact_path in artifact_paths:
            path = Path(artifact_path)
            if path.exists():
                mlflow.log_artifact(str(path))
    return True
