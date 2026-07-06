"""Generate grounded stakeholder reports from synthetic experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "experiments"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports"


def _fmt(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def load_artifacts(comparison_id: str, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> tuple[pd.DataFrame, Dict[str, object]]:
    comparison_dir = artifact_dir / comparison_id
    kpis_path = comparison_dir / "kpis.csv"
    profile_path = comparison_dir / "synthetic_profile.json"
    if not kpis_path.exists():
        raise FileNotFoundError(f"{kpis_path} not found")
    if not profile_path.exists():
        raise FileNotFoundError(f"{profile_path} not found")
    kpis = pd.read_csv(kpis_path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    return kpis, profile


def build_report(comparison_id: str, kpis: pd.DataFrame, profile: Dict[str, object]) -> str:
    required = {
        "run_id",
        "rung",
        "nr_mode",
        "policy_quality_score",
        "flights_delay_dep",
        "interval_spillage",
        "nr_uncovered_labor_hours",
        "completion_factor",
    }
    missing = sorted(required - set(kpis.columns))
    if missing:
        raise ValueError(f"Missing KPI columns: {missing}")

    best = kpis.sort_values("policy_quality_score", ascending=False).iloc[0]
    delay_best = kpis.sort_values("flights_delay_dep", ascending=True).iloc[0]
    nr_best = kpis.sort_values("nr_uncovered_labor_hours", ascending=True).iloc[0]

    static_mean = kpis[kpis["nr_mode"] == "static"]["policy_quality_score"].mean()
    predicted_mean = kpis[kpis["nr_mode"] == "predicted"]["policy_quality_score"].mean()
    predicted_delta = predicted_mean - static_mean

    lines: List[str] = [
        "# Grounded Experiment Analyst Report",
        "",
        f"Comparison: `{comparison_id}`",
        "",
        "## Scenario Evidence",
        "",
        (
            f"The synthetic scenario contains {profile['aircraft_count']} aircraft, "
            f"{profile['airport_count']} airports, {profile['rotation_count']} rotations, "
            f"and {profile['flight_leg_count']} flight legs over {profile['duration_days']} days."
        ),
        "",
        "## Main Finding",
        "",
        (
            f"The strongest proxy policy is `{best['rung']}` with `{best['nr_mode']}` NR mode. "
            f"It reaches a policy quality score of {_fmt(best['policy_quality_score'])} "
            f"in run `{best['run_id']}`."
        ),
        "",
        "## KPI Evidence",
        "",
        (
            f"- Lowest departure delay: `{delay_best['run_id']}` with "
            f"{_fmt(delay_best['flights_delay_dep'])} total delay proxy units."
        ),
        (
            f"- Lowest uncovered NR labour: `{nr_best['run_id']}` with "
            f"{_fmt(nr_best['nr_uncovered_labor_hours'])} hours."
        ),
        (
            f"- Best policy interval spillage: `{best['run_id']}` with "
            f"{_fmt(best['interval_spillage'])}%."
        ),
        (
            f"- Best policy completion factor: `{best['run_id']}` with "
            f"{_fmt(best['completion_factor'], 4)}."
        ),
        "",
        "## NR Mode Comparison",
        "",
        (
            f"Across tracked runs, predicted NR mode changes the mean policy quality score by "
            f"{_fmt(predicted_delta)} versus static mode "
            f"({_fmt(predicted_mean)} predicted vs {_fmt(static_mean)} static)."
        ),
        "",
        "## Caveat",
        "",
        (
            "This report is generated only from synthetic experiment artifacts. It is useful for "
            "reviewing workflow behavior and policy-comparison plumbing, not for making claims "
            "about a real airline operation."
        ),
        "",
    ]
    return "\n".join(lines)


def write_report(
    comparison_id: str,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> Path:
    kpis, profile = load_artifacts(comparison_id, artifact_dir)
    report = build_report(comparison_id, kpis, profile)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{comparison_id}_analyst_report.md"
    path.write_text(report, encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a grounded analyst report from experiment artifacts.")
    parser.add_argument("comparison_id", help="Comparison artifact folder, e.g. default_run_comparison_seed20260706")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    path = write_report(args.comparison_id, Path(args.artifact_dir), Path(args.report_dir))
    print(f"Wrote grounded analyst report to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
