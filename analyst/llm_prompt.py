"""Build an LLM-ready prompt package from grounded experiment evidence.

This module does not call a model. It prepares a strict evidence bundle that can be
sent to any LLM provider later. Keeping generation separate from API calls makes the
public repo reproducible and prevents ungrounded claims from entering reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from analyst.experiment_report import DEFAULT_ARTIFACT_DIR, DEFAULT_REPORT_DIR, load_artifacts, write_report


DEFAULT_PROMPT_DIR = Path(__file__).resolve().parents[1] / "reports" / "llm_prompts"


def _records_for_prompt(kpis: pd.DataFrame) -> List[Dict[str, object]]:
    cols = [
        "run_id",
        "rung",
        "nr_mode",
        "policy_quality_score",
        "flights_delay_dep",
        "interval_spillage",
        "nr_uncovered_labor_hours",
        "completion_factor",
    ]
    available = [col for col in cols if col in kpis.columns]
    return kpis[available].round(4).to_dict("records")


def build_prompt_package(
    comparison_id: str,
    report_markdown: str,
    kpis: pd.DataFrame,
    profile: Dict[str, object],
) -> Dict[str, object]:
    """Create a provider-neutral LLM prompt package with hard grounding rules."""

    evidence = {
        "comparison_id": comparison_id,
        "scenario_profile": profile,
        "kpi_records": _records_for_prompt(kpis),
        "grounded_report": report_markdown,
    }
    return {
        "comparison_id": comparison_id,
        "system": (
            "You are an experiment analyst. Rewrite the supplied grounded report for a "
            "technical stakeholder. Use only the evidence JSON. Preserve all run IDs and "
            "metric values you mention. Do not infer real-world operational performance. "
            "If a claim is not supported by the evidence, omit it."
        ),
        "user": (
            "Create a concise stakeholder summary with sections: Executive summary, "
            "Policy comparison, NR workload evidence, Caveats. Every quantitative claim "
            "must cite a run_id or scenario profile field from the evidence."
        ),
        "evidence": evidence,
    }


def write_prompt_package(
    comparison_id: str,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
    prompt_dir: Path = DEFAULT_PROMPT_DIR,
) -> Path:
    report_path = report_dir / f"{comparison_id}_analyst_report.md"
    if not report_path.exists():
        write_report(comparison_id, artifact_dir, report_dir)
    report_markdown = report_path.read_text(encoding="utf-8")
    kpis, profile = load_artifacts(comparison_id, artifact_dir)
    package = build_prompt_package(comparison_id, report_markdown, kpis, profile)

    prompt_dir.mkdir(parents=True, exist_ok=True)
    path = prompt_dir / f"{comparison_id}_llm_prompt.json"
    path.write_text(json.dumps(package, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an LLM-ready grounded prompt package.")
    parser.add_argument("comparison_id", help="Comparison artifact folder, e.g. default_run_comparison_seed20260706")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--prompt-dir", default=str(DEFAULT_PROMPT_DIR))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    path = write_prompt_package(
        args.comparison_id,
        Path(args.artifact_dir),
        Path(args.report_dir),
        Path(args.prompt_dir),
    )
    print(f"Wrote grounded LLM prompt package to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
