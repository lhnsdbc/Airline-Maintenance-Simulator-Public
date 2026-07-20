"""Deterministic synthetic policy-comparison experiment.

This module gives the public repo a fast, reproducible experiment-tracking workflow
without requiring the full optimizer stack. It reads the generated synthetic fixtures,
computes scale-aware proxy KPIs for policy rungs and NR modes, and writes one local
experiment record per policy/mode combination.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd

from experiments.tracking import ExperimentRecorder, try_log_mlflow, write_json
from experiments.rl_llm_evaluation import build_evaluation_artifact, write_evaluation_artifact


RUNG_FACTORS = {
    "R0": {"label": "strict_baseline", "delay": 1.00, "spillage": 1.00, "maintenance": 1.00},
    "R1": {"label": "constrained_optimizer", "delay": 0.88, "spillage": 0.76, "maintenance": 0.91},
    "R2": {"label": "free_optimizer", "delay": 0.80, "spillage": 0.63, "maintenance": 0.84},
}
NR_MODES = ("static", "predicted")


def _stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256("|".join([str(seed), *parts]).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `py generate_dummy_data.py` and "
            "`py generate_mock_nr_artifact.py` first."
        )
    return path


def load_profile(data_dir: Path, scenario: str) -> Dict[str, object]:
    input_dir = data_dir / "input"
    schedule_path = _require(input_dir / "schedules" / "schedule_2023-01-01_1weeks")
    with schedule_path.open("rb") as f:
        schedule = pickle.load(f)

    scenarios = pd.read_csv(_require(input_dir / "Scenarios_simulation.csv"))
    scenario_row = scenarios[scenarios["Id"] == scenario]
    if scenario_row.empty:
        raise ValueError(f"Scenario {scenario!r} not found in {input_dir / 'Scenarios_simulation.csv'}")

    aircraft = pd.read_csv(_require(input_dir / "AircraftRegistrations.csv"))
    airports = pd.read_csv(_require(input_dir / "Airports.csv"))
    slots = pd.read_excel(_require(input_dir / "Slots_norm_scenarios.xlsx"))
    policy = pd.read_csv(_require(input_dir / "sac_gnn" / "Maintenance policy data.csv"))
    nr = pd.read_csv(_require(input_dir / "nr_prediction" / "nr_conditional.csv"))
    nr_task_rows = nr[nr["rt_code"] != "__fleet_default__"]

    duration_days = 7
    block_hours = float(schedule["ActualBlockTimeDuration"].sum() / 60.0)
    profile = {
        "scenario": scenario,
        "duration_days": duration_days,
        "aircraft_count": int(len(aircraft)),
        "airport_count": int(len(airports)),
        "rotation_count": int(schedule["RotationId"].nunique()),
        "flight_leg_count": int(len(schedule)),
        "maintenance_slot_templates": int(len(slots)),
        "hangar_a_slot_templates": int((slots["Slot_type"] == "A").sum()),
        "policy_task_count": int(len(policy)),
        "nr_task_count": int(len(nr_task_rows)),
        "total_block_hours": round(block_hours, 3),
        "fleet_utilization": round(block_hours / (len(aircraft) * duration_days * 24), 4),
        "mean_flight_duration_min": round(float(schedule["ActualBlockTimeDuration"].mean()), 3),
        "median_task_labor_hours": round(float(policy["Labour"].median()), 3),
        "total_task_labor_hours": round(float(policy["Labour"].sum()), 3),
        "mean_nr_q50_hours": round(float(nr_task_rows["q50"].mean()), 3),
        "mean_nr_probability": round(float(nr_task_rows["p_nr"].mean()), 4),
    }
    return profile


def compute_metrics(profile: Mapping[str, object], rung: str, nr_mode: str, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(_stable_seed(seed, rung, nr_mode))
    factors = RUNG_FACTORS[rung]
    legs = float(profile["flight_leg_count"])
    rotations = float(profile["rotation_count"])
    utilization = float(profile["fleet_utilization"])
    task_labor = float(profile["total_task_labor_hours"])
    nr_expected = float(profile["mean_nr_q50_hours"]) * float(profile["mean_nr_probability"])
    hangar_slots = float(profile["hangar_a_slot_templates"])

    predicted_bonus = 0.78 if nr_mode == "predicted" else 1.0
    reserve_bias = 1.12 if nr_mode == "predicted" else 0.94
    noise = float(rng.normal(0.0, 0.015))

    delay_per_leg = max(0.0, 18.0 * float(factors["delay"]) * predicted_bonus * (1.0 + noise))
    nr_realized = nr_expected * task_labor * 0.018 * (1.0 + float(rng.normal(0.0, 0.04)))
    nr_reserved = nr_realized * reserve_bias * (1.0 + float(rng.normal(0.0, 0.025)))
    nr_uncovered = max(0.0, nr_realized - nr_reserved)

    metrics = {
        "flight_leg_count": legs,
        "rotation_count": rotations,
        "fleet_utilization": utilization,
        "flights_delay_dep": round(legs * delay_per_leg, 3),
        "rotations_delay_dep": round(rotations * delay_per_leg * 1.35, 3),
        "completion_factor": round(max(0.0, 1.0 - delay_per_leg / 900.0), 4),
        "interval_spillage": round(28.0 * float(factors["spillage"]) * (1.0 + noise), 3),
        "tasks_execution_factor": round(min(1.0, 0.88 + (1.0 - float(factors["spillage"])) * 0.18), 4),
        "a_check_slots_executed": round(hangar_slots * float(factors["maintenance"]), 3),
        "nr_reserved_hours": round(nr_reserved, 3),
        "nr_realized_hours": round(nr_realized, 3),
        "nr_uncovered_labor_hours": round(nr_uncovered, 3),
        "nr_reserve_realized_corr": round(0.62 if nr_mode == "predicted" else 0.28, 3),
        "policy_quality_score": round(
            100.0
            - 0.015 * legs * delay_per_leg
            - 0.85 * 28.0 * float(factors["spillage"])
            - 1.7 * nr_uncovered,
            3,
        ),
    }
    return metrics


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary_markdown(profile: Mapping[str, object], rows: List[Mapping[str, object]]) -> str:
    best = max(rows, key=lambda row: float(row["policy_quality_score"]))
    return "\n".join(
        [
            "# Synthetic Policy Comparison",
            "",
            f"Scenario `{profile['scenario']}` uses {profile['aircraft_count']} aircraft, "
            f"{profile['airport_count']} airports, {profile['rotation_count']} rotations, "
            f"and {profile['flight_leg_count']} flight legs.",
            "",
            f"Best proxy policy in this deterministic workflow: `{best['rung']}` with "
            f"`{best['nr_mode']}` NR mode, quality score {best['policy_quality_score']}.",
            "",
            "This is a synthetic validation experiment. It demonstrates run tracking and "
            "policy-comparison plumbing; it is not evidence about a real airline operation.",
            "",
        ]
    )


def run_experiment(args: argparse.Namespace) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "Data"
    output_dir = repo_root / args.output_dir
    profile = load_profile(data_dir, args.scenario)

    revision = _git_revision()
    recorder = ExperimentRecorder(output_dir)
    comparison_rows: List[Dict[str, object]] = []
    run_artifact_dirs: List[Path] = []

    for rung in RUNG_FACTORS:
        for nr_mode in NR_MODES:
            run_id = f"{args.scenario}_{rung}_{nr_mode}_seed{args.seed}"
            params = {
                "scenario_id": args.scenario,
                "random_seed": args.seed,
                "simulator_version": revision,
                "policy_rung": rung,
                "policy_label": RUNG_FACTORS[rung]["label"],
                "nr_mode": nr_mode,
                "synthetic_profile_version": "shape_matched_v1",
            }
            metrics = compute_metrics(profile, rung, nr_mode, args.seed)
            run_path = recorder.write_record(run_id, params=params, metrics=metrics, artifacts={})
            run_artifact_dirs.append(run_path)
            comparison_rows.append({"run_id": run_id, "rung": rung, "nr_mode": nr_mode, **metrics})
            try_log_mlflow(
                experiment_name="synthetic-policy-comparison",
                run_name=run_id,
                params=params,
                metrics=metrics,
                artifact_paths=[run_path / "metadata.json", run_path / "metrics.json"],
            )

    summary_dir = output_dir / f"{args.scenario}_comparison_seed{args.seed}"
    summary_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(summary_dir / "kpis.csv", comparison_rows)
    write_json(summary_dir / "synthetic_profile.json", profile)
    (summary_dir / "summary.md").write_text(_summary_markdown(profile, comparison_rows), encoding="utf-8")
    evaluation_artifact = build_evaluation_artifact(
        summary_dir.name,
        pd.DataFrame(comparison_rows),
        profile,
        revision,
    )
    write_evaluation_artifact(summary_dir, evaluation_artifact)
    best = max(comparison_rows, key=lambda row: float(row["policy_quality_score"]))
    summary_metrics = {
        "best_policy_quality_score": best["policy_quality_score"],
        "best_departure_delay": min(float(row["flights_delay_dep"]) for row in comparison_rows),
        "best_uncovered_nr_labor_hours": min(float(row["nr_uncovered_labor_hours"]) for row in comparison_rows),
        "run_count": len(comparison_rows),
    }
    summary_params = {
        "scenario_id": args.scenario,
        "random_seed": args.seed,
        "simulator_version": revision,
        "best_policy_rung": best["rung"],
        "best_nr_mode": best["nr_mode"],
        "tracking_scope": "comparison_summary",
    }
    summary_artifacts = [
        summary_dir / "kpis.csv",
        summary_dir / "synthetic_profile.json",
        summary_dir / "summary.md",
        summary_dir / "rl_llm_evaluation.json",
    ]
    summary_logged_to_mlflow = try_log_mlflow(
        experiment_name="synthetic-policy-comparison",
        run_name=summary_dir.name,
        params=summary_params,
        metrics=summary_metrics,
        artifact_paths=summary_artifacts,
    )
    write_json(summary_dir / "mlflow_manifest.json", {
        "experiment_name": "synthetic-policy-comparison",
        "comparison_id": summary_dir.name,
        "mlflow_logged": summary_logged_to_mlflow,
        "install_extra": "pip install -r requirements-mlops.txt",
        "params": summary_params,
        "metrics": summary_metrics,
        "run_artifact_dirs": [str(path) for path in run_artifact_dirs],
        "summary_artifacts": [str(path) for path in summary_artifacts],
    })
    return summary_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run tracked synthetic policy-comparison workflow.")
    parser.add_argument("--scenario", default="default_run")
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--output-dir", default="artifacts/experiments")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary_dir = run_experiment(args)
    print(f"Wrote synthetic experiment records to {summary_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
