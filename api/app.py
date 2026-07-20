"""FastAPI wrapper for synthetic simulator experiment workflows."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from analyst.live_llm import generate_grounded_llm_report
from experiments.synthetic_experiment import load_profile, run_experiment
from retrieval.search import search_artifacts
from retrieval.vector import DEFAULT_VECTOR_DIR, vector_search


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "Data"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "experiments"
PIPELINE_STATUS_PATH = REPO_ROOT / "artifacts" / "data_lake" / "gold" / "pipeline_status" / "latest.json"


class ComparePoliciesRequest(BaseModel):
    scenario: str = Field(default="default_run", min_length=1, max_length=80)
    seed: int = Field(default=20260706, ge=0, le=2_147_483_647)


class ExperimentSummary(BaseModel):
    comparison_id: str
    path: str
    kpi_rows: int


class SearchResponse(BaseModel):
    query: str
    count: int
    results: List[Dict[str, Any]]


class LlmReportRequest(BaseModel):
    provider: str = Field(default="deterministic", pattern="^(auto|deterministic|openai|anthropic|gemini)$")


class LlmReportResponse(BaseModel):
    comparison_id: str
    provider: str
    model: str
    used_live_provider: bool
    report_path: str
    metadata_path: str
    text: str


class RuntimeMetrics:
    def __init__(self) -> None:
        self.started_at = time.time()
        self.request_count = 0
        self.failed_request_count = 0
        self.generated_comparison_count = 0
        self.search_count = 0
        self.llm_report_count = 0
        self.total_latency_seconds = 0.0

    def record(self, latency_seconds: float, failed: bool) -> None:
        self.request_count += 1
        self.total_latency_seconds += latency_seconds
        if failed:
            self.failed_request_count += 1

    def snapshot(self) -> Dict[str, Any]:
        avg_latency = (
            self.total_latency_seconds / self.request_count
            if self.request_count
            else 0.0
        )
        return {
            "uptime_seconds": round(time.time() - self.started_at, 3),
            "request_count": self.request_count,
            "failed_request_count": self.failed_request_count,
            "generated_comparison_count": self.generated_comparison_count,
            "search_count": self.search_count,
            "llm_report_count": self.llm_report_count,
            "average_latency_seconds": round(avg_latency, 6),
            "artifact_comparison_count": len(_comparison_dirs()),
        }


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


def _pipeline_status() -> Dict[str, Any]:
    connection_string = os.getenv("PIPELINE_STORAGE_CONNECTION_STRING")
    if connection_string:
        try:
            from azure.storage.blob import BlobServiceClient

            container = BlobServiceClient.from_connection_string(connection_string).get_container_client(
                os.getenv("PIPELINE_FILE_SYSTEM", "maintenance-lake")
            )
            return json.loads(container.download_blob("gold/pipeline_status/latest.json").readall())
        except Exception:
            pass
    if not PIPELINE_STATUS_PATH.exists():
        return {"status": "not_run"}
    return _read_json(PIPELINE_STATUS_PATH)


def create_app() -> FastAPI:
    runtime_metrics = RuntimeMetrics()
    app = FastAPI(
        title="Aircraft Maintenance ML Simulator API",
        version="0.1.0",
        description="Public synthetic-data API for simulator experiment tracking and policy comparison.",
    )

    @app.middleware("http")
    async def record_request_metrics(request: Request, call_next):
        start = time.time()
        failed = False
        try:
            response = await call_next(request)
            failed = response.status_code >= 500
            return response
        except Exception:
            failed = True
            raise
        finally:
            runtime_metrics.record(time.time() - start, failed)

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "deployment_version": os.getenv("APP_VERSION", "local"),
            "data_generated": (DATA_DIR / "input").exists(),
            "artifact_dir": str(ARTIFACT_DIR),
            "pipeline_status": _pipeline_status()["status"],
        }

    @app.get("/")
    def service_index() -> Dict[str, Any]:
        return {
            "name": "Aircraft Maintenance ML Simulator API",
            "deployment_version": os.getenv("APP_VERSION", "local"),
            "scope": "public synthetic-data demo",
            "docs_url": "/docs",
            "health_url": "/health",
            "metrics_url": "/metrics",
            "experiments_url": "/experiments",
            "lexical_search_example": "/search?q=predicted%20uncovered&nr_mode=predicted",
            "rag_search_example": "/rag/search?q=predicted%20uncovered&nr_mode=predicted",
            "llm_report_example": "/experiments/default_run_comparison_seed20260706/llm-report",
            "rl_llm_evaluation_example": "/experiments/default_run_comparison_seed20260706/rl-llm-evaluation",
            "pipeline_status_url": "/pipeline-status",
        }

    @app.get("/pipeline-status")
    def pipeline_status() -> Dict[str, Any]:
        return _pipeline_status()

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
        runtime_metrics.generated_comparison_count += 1
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

    @app.get("/experiments/{comparison_id}/rl-llm-evaluation")
    def experiment_rl_llm_evaluation(comparison_id: str) -> Dict[str, Any]:
        path = ARTIFACT_DIR / comparison_id / "rl_llm_evaluation.json"
        if not path.resolve().is_relative_to(ARTIFACT_DIR.resolve()):
            raise HTTPException(status_code=400, detail="Invalid comparison id")
        return _read_json(path)

    @app.get("/search", response_model=SearchResponse)
    def search(
        q: str,
        comparison_id: Optional[str] = None,
        rung: Optional[str] = None,
        nr_mode: Optional[str] = None,
        kind: Optional[str] = None,
        limit: int = 5,
    ) -> SearchResponse:
        if limit < 1 or limit > 50:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 50")
        results = search_artifacts(
            q,
            artifact_dir=ARTIFACT_DIR,
            report_dir=REPO_ROOT / "reports",
            comparison_id=comparison_id,
            rung=rung,
            nr_mode=nr_mode,
            kind=kind,
            limit=limit,
        )
        runtime_metrics.search_count += 1
        return SearchResponse(query=q, count=len(results), results=results)

    @app.get("/rag/search", response_model=SearchResponse)
    def rag_search(
        q: str,
        backend: str = "local",
        comparison_id: Optional[str] = None,
        rung: Optional[str] = None,
        nr_mode: Optional[str] = None,
        kind: Optional[str] = None,
        limit: int = 5,
    ) -> SearchResponse:
        if limit < 1 or limit > 50:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 50")
        try:
            results = vector_search(
                q,
                artifact_dir=ARTIFACT_DIR,
                report_dir=REPO_ROOT / "reports",
                index_dir=DEFAULT_VECTOR_DIR,
                backend=backend,
                comparison_id=comparison_id,
                rung=rung,
                nr_mode=nr_mode,
                kind=kind,
                limit=limit,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        runtime_metrics.search_count += 1
        return SearchResponse(query=q, count=len(results), results=results)

    @app.post("/experiments/{comparison_id}/llm-report", response_model=LlmReportResponse)
    def llm_report(comparison_id: str, request: LlmReportRequest) -> LlmReportResponse:
        comparison_path = ARTIFACT_DIR / comparison_id
        if not comparison_path.resolve().is_relative_to(ARTIFACT_DIR.resolve()):
            raise HTTPException(status_code=400, detail="Invalid comparison id")
        if not comparison_path.exists():
            raise HTTPException(status_code=404, detail="comparison not found")
        try:
            result = generate_grounded_llm_report(
                comparison_id,
                provider=request.provider,
                artifact_dir=ARTIFACT_DIR,
                report_dir=REPO_ROOT / "reports",
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        runtime_metrics.llm_report_count += 1
        return LlmReportResponse(
            comparison_id=comparison_id,
            provider=result.provider,
            model=result.model,
            used_live_provider=result.used_live_provider,
            report_path=str(result.output_path),
            metadata_path=str(result.metadata_path),
            text=result.text,
        )

    @app.get("/metrics")
    def metrics() -> Dict[str, Any]:
        return runtime_metrics.snapshot()

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
