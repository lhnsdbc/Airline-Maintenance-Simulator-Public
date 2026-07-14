"""Verify deployed API and dashboard URLs for the public demo."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, cast


DEFAULT_API_URL = "https://maintenance-simulator-api.onrender.com"
DEFAULT_DASHBOARD_URL = "https://maintenance-simulator-dashboard.onrender.com"
COMPARISON_ID = "default_run_comparison_seed20260706"


def _join(base_url: str, path: str) -> str:
    return urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def _request_json(url: str, *, method: str = "GET", payload: Dict[str, Any] | None = None, timeout: float = 20.0) -> Dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return cast(Dict[str, Any], json.loads(response.read().decode("utf-8")))


def _request_status(url: str, *, timeout: float = 20.0) -> int:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return int(response.status)


def _retry(label: str, fn, attempts: int, delay_seconds: float):
    last_error = None
    for _ in range(attempts):
        try:
            return fn()
        except (urllib.error.URLError, TimeoutError, AssertionError) as exc:
            last_error = exc
            time.sleep(delay_seconds)
    raise SystemExit(f"{label} failed after {attempts} attempts: {last_error}")


def verify(api_url: str, dashboard_url: str, attempts: int = 6, delay_seconds: float = 10.0) -> Dict[str, Any]:
    api_index = _retry("API index", lambda: _request_json(_join(api_url, "/")), attempts, delay_seconds)
    assert api_index["scope"] == "public synthetic-data demo"

    api_health = _retry("API health", lambda: _request_json(_join(api_url, "/health")), attempts, delay_seconds)
    assert api_health["status"] == "ok"

    metrics = _retry("API metrics", lambda: _request_json(_join(api_url, "/metrics")), attempts, delay_seconds)
    assert "request_count" in metrics

    experiments = _retry("API experiments", lambda: _request_json(_join(api_url, "/experiments")), attempts, delay_seconds)
    assert any(item["comparison_id"] == COMPARISON_ID for item in experiments)

    kpis = _retry(
        "API KPI contract",
        lambda: _request_json(_join(api_url, f"/experiments/{COMPARISON_ID}/kpis")),
        attempts,
        delay_seconds,
    )
    assert kpis["comparison_id"] == COMPARISON_ID
    assert isinstance(kpis["rows"], list) and len(kpis["rows"]) >= 6
    required_kpi_fields = {"run_id", "rung", "nr_mode", "policy_quality_score", "flights_delay_dep"}
    assert required_kpi_fields.issubset(kpis["rows"][0])

    rag_path = "/rag/search?q=predicted%20uncovered&nr_mode=predicted"
    rag = _retry("RAG search", lambda: _request_json(_join(api_url, rag_path)), attempts, delay_seconds)
    assert rag["count"] >= 1
    assert all(result["metadata"].get("nr_mode") == "predicted" for result in rag["results"])

    llm_path = f"/experiments/{COMPARISON_ID}/llm-report"
    llm = _retry(
        "LLM report",
        lambda: _request_json(_join(api_url, llm_path), method="POST", payload={"provider": "deterministic"}),
        attempts,
        delay_seconds,
    )
    assert llm["provider"] == "deterministic"
    assert llm["used_live_provider"] is False

    dashboard_health = _retry("Dashboard health", lambda: _request_json(_join(dashboard_url, "/health")), attempts, delay_seconds)
    assert dashboard_health["status"] == "ok"
    assert dashboard_health["row_count"] >= 6

    dashboard_status = _retry("Dashboard root", lambda: _request_status(_join(dashboard_url, "/")), attempts, delay_seconds)
    assert dashboard_status == 200

    return {
        "api_url": api_url,
        "dashboard_url": dashboard_url,
        "api_status": api_health["status"],
        "dashboard_status": dashboard_health["status"],
        "experiment_count": len(experiments),
        "kpi_row_count": len(kpis["rows"]),
        "rag_result_count": rag["count"],
        "llm_provider": llm["provider"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify deployed public demo services.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--dashboard-url", default=DEFAULT_DASHBOARD_URL)
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--delay-seconds", type=float, default=10.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = verify(args.api_url, args.dashboard_url, args.attempts, args.delay_seconds)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
