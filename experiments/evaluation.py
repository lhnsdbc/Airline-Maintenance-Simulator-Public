"""Repeatable synthetic evaluation for the public policy-comparison demo.

This module deliberately evaluates *proxies*, not a trained production policy.  It
keeps the same generated fixture and applies named deterministic stress profiles so
the public repository can demonstrate an evaluation protocol without implying that
the results describe an airline operation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np

from experiments.synthetic_experiment import RUNG_FACTORS, compute_metrics, load_profile


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (20260706, 20260707, 20260708, 20260709, 20260710)
EVALUATION_SCENARIOS = {
    "nominal": {
        "description": "Generated one-week fixture without an added stress multiplier.",
        "delay_multiplier": 1.00,
        "spillage_multiplier": 1.00,
        "slot_multiplier": 1.00,
    },
    "high_utilization": {
        "description": "Synthetic demand-pressure profile with higher delay and spillage proxies.",
        "delay_multiplier": 1.18,
        "spillage_multiplier": 1.08,
        "slot_multiplier": 0.95,
    },
    "constrained_hangar": {
        "description": "Synthetic maintenance-capacity profile with fewer usable A-check slots.",
        "delay_multiplier": 1.08,
        "spillage_multiplier": 1.28,
        "slot_multiplier": 0.84,
    },
}
POLICIES = {
    "strict_baseline": "R0",
    "learned_policy_proxy": "R2",
}


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def apply_scenario(metrics: Mapping[str, float], scenario: str) -> Dict[str, float]:
    """Apply a documented synthetic stress profile and recompute the composite proxy."""
    if scenario not in EVALUATION_SCENARIOS:
        raise ValueError(f"Unknown evaluation scenario: {scenario}")
    factors = EVALUATION_SCENARIOS[scenario]
    adjusted = dict(metrics)
    adjusted["flights_delay_dep"] = round(float(metrics["flights_delay_dep"]) * factors["delay_multiplier"], 3)
    adjusted["interval_spillage"] = round(float(metrics["interval_spillage"]) * factors["spillage_multiplier"], 3)
    adjusted["a_check_slots_executed"] = round(float(metrics["a_check_slots_executed"]) * factors["slot_multiplier"], 3)
    adjusted["policy_quality_score"] = round(
        100.0
        - 0.015 * adjusted["flights_delay_dep"]
        - 0.85 * adjusted["interval_spillage"]
        - 1.7 * float(adjusted["nr_uncovered_labor_hours"]),
        3,
    )
    return adjusted


def evaluate(profile: Mapping[str, object], seeds: Iterable[int] = DEFAULT_SEEDS) -> List[Dict[str, object]]:
    """Evaluate baseline and learning-inspired proxy with common NR mode across fixed seeds."""
    rows: List[Dict[str, object]] = []
    for scenario in EVALUATION_SCENARIOS:
        for seed in seeds:
            for policy, rung in POLICIES.items():
                metrics = apply_scenario(compute_metrics(profile, rung, "predicted", seed), scenario)
                rows.append({
                    "scenario": scenario,
                    "seed": seed,
                    "policy": policy,
                    "rung": rung,
                    "nr_mode": "predicted",
                    **metrics,
                })
    return rows


def summarize(rows: Iterable[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Return median and P10--P90 distributions for the evaluation card."""
    rows = list(rows)
    summary: List[Dict[str, object]] = []
    for scenario in EVALUATION_SCENARIOS:
        for policy in POLICIES:
            selected = [row for row in rows if row["scenario"] == scenario and row["policy"] == policy]
            for metric in ("policy_quality_score", "flights_delay_dep", "interval_spillage", "a_check_slots_executed"):
                values = np.array([float(row[metric]) for row in selected])
                summary.append({
                    "scenario": scenario,
                    "policy": policy,
                    "metric": metric,
                    "n": len(values),
                    "median": round(float(np.median(values)), 3),
                    "p10": round(float(np.quantile(values, 0.10)), 3),
                    "p90": round(float(np.quantile(values, 0.90)), 3),
                })
    return summary


def run(output_dir: Path, seeds: Iterable[int] = DEFAULT_SEEDS) -> Path:
    seeds = tuple(seeds)
    profile = load_profile(REPO_ROOT / "Data", "default_run")
    rows = evaluate(profile, seeds)
    _write_csv(output_dir / "runs.csv", rows)
    _write_csv(output_dir / "summary.csv", summarize(rows))
    (output_dir / "protocol.json").write_text(
        json.dumps({
            "scope": "synthetic policy-proxy evaluation; not real operational performance",
            "fixture_scenario": "default_run",
            "seeds": list(seeds),
            "nr_mode": "predicted (held constant across policies)",
            "scenarios": EVALUATION_SCENARIOS,
            "policies": {name: {"rung": rung, "source_label": RUNG_FACTORS[rung]["label"]} for name, rung in POLICIES.items()},
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the repeatable synthetic evaluation card protocol.")
    parser.add_argument("--output-dir", default="artifacts/evaluation_card")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = run(REPO_ROOT / args.output_dir, args.seeds)
    print(f"Wrote synthetic evaluation card artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
