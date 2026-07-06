"""FastAPI wrapper for synthetic simulator experiment workflows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from experiments.synthetic_experiment import load_profile, run_experiment


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "Data"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "experiments"


class ComparePoliciesRequest(BaseModel):
    scenario: str = Field(default="default_run", min_length=1, max_length=80)
    seed: int = Field(default=20260706, ge=0, le=2_147_483_647)


class ExperimentSummary(BaseModel):
    comparison_id: str
    path: str
    kpi_rows: int


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"{path.name} not found")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid JSON artifact: {exc}")


def _comparison_dirs() -> List[Path]:
    if not ARTIFACT_DIR.exists():
        return []
    return sorted(ARTIFACT_DIR.glob("*_comparison_seed*"))


def create_app() -> FastAPI:
    app = FastAPI(
        title="Aircraft Maintenance ML Simulator API",
        version="0.1.0",
        description="Public synthetic-data API for simulator experiment tracking and policy comparison.",
    )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "data_generated": (DATA_DIR / "input").exists(),
            "artifact_dir": str(ARTIFACT_DIR),
        }

    @app.get("/profile/{scenario}")
    def profile(scenario: str) -> Dict[str, Any]:
        try:
            return load_profile(DATA_DIR, scenario)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.post("/compare-policies")
    def compare_policies(request: ComparePoliciesRequest) -> Dict[str, Any]:
        args = argparse.Namespace(
            scenario=request.scenario,
            seed=request.seed,
            output_dir=str(ARTIFACT_DIR.relative_to(REPO_ROOT)),
        )
        try:
            summary_dir = run_experiment(args)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {
            "comparison_id": summary_dir.name,
            "scenario": request.scenario,
            "seed": request.seed,
            "kpis_url": f"/experiments/{summary_dir.name}/kpis",
            "metadata_url": f"/experiments/{summary_dir.name}/profile",
        }

    @app.get("/experiments", response_model=List[ExperimentSummary])
    def experiments() -> List[ExperimentSummary]:
        summaries = []
        for path in _comparison_dirs():
            kpis_path = path / "kpis.csv"
            if not kpis_path.exists():
                continue
            rows = max(0, len(kpis_path.read_text(encoding="utf-8").splitlines()) - 1)
            summaries.append(ExperimentSummary(
                comparison_id=path.name,
                path=str(path),
                kpi_rows=rows,
            ))
        return summaries

    @app.get("/experiments/{comparison_id}/profile")
    def experiment_profile(comparison_id: str) -> Dict[str, Any]:
        path = ARTIFACT_DIR / comparison_id / "synthetic_profile.json"
        if not path.resolve().is_relative_to(ARTIFACT_DIR.resolve()):
            raise HTTPException(status_code=400, detail="Invalid comparison id")
        return _read_json(path)

    @app.get("/experiments/{comparison_id}/kpis")
    def experiment_kpis(comparison_id: str) -> Dict[str, Any]:
        path = ARTIFACT_DIR / comparison_id / "kpis.csv"
        if not path.resolve().is_relative_to(ARTIFACT_DIR.resolve()):
            raise HTTPException(status_code=400, detail="Invalid comparison id")
        if not path.exists():
            raise HTTPException(status_code=404, detail="kpis.csv not found")
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return {"comparison_id": comparison_id, "rows": rows}

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
