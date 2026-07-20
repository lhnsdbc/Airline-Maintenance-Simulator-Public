"""Scope-labelled RL/LLM evaluation artifacts for synthetic policy comparisons.

This module deliberately separates deterministic, verifiable rewards from an
optional LLM-as-judge protocol.  It does not train or serve a language model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def _bounded_quality(value: float, ceiling: float) -> float:
    return round(max(0.0, min(1.0, 1.0 - value / ceiling)), 4)


def build_reward_audit(
    row: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Create deterministic reward components and checks from simulator KPIs."""

    flight_legs = max(float(profile["flight_leg_count"]), 1.0)
    uncovered = float(row["nr_uncovered_labor_hours"])
    completion = float(row["completion_factor"])
    delay = float(row["flights_delay_dep"])
    spillage = float(row["interval_spillage"])
    policy_quality = float(row["policy_quality_score"])

    components = {
        "service_completion": round(max(0.0, min(1.0, completion)), 4),
        "delay_quality": _bounded_quality(delay, flight_legs * 20.0),
        "spillage_quality": _bounded_quality(spillage, 30.0),
        "uncovered_labor_quality": _bounded_quality(uncovered, 4.0),
        "policy_quality": round(max(0.0, min(1.0, policy_quality / 100.0)), 4),
    }
    reward = round(
        0.25 * components["service_completion"]
        + 0.25 * components["delay_quality"]
        + 0.20 * components["spillage_quality"]
        + 0.20 * components["uncovered_labor_quality"]
        + 0.10 * components["policy_quality"],
        4,
    )
    checks = {
        "no_uncovered_nr_labor": {
            "passed": uncovered <= 0.001,
            "observed": round(uncovered, 4),
            "threshold": 0.001,
        },
        "minimum_completion_factor": {
            "passed": completion >= 0.97,
            "observed": round(completion, 4),
            "threshold": 0.97,
        },
        "bounded_interval_spillage": {
            "passed": spillage <= 30.0,
            "observed": round(spillage, 4),
            "threshold": 30.0,
        },
    }
    return {
        "reward": reward,
        "components": components,
        "verifiable_checks": checks,
        "all_checks_passed": all(check["passed"] for check in checks.values()),
    }


def build_evaluation_artifact(
    comparison_id: str,
    kpis: pd.DataFrame,
    profile: Mapping[str, Any],
    simulator_version: str,
    moe_training: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an inspectable design artifact without implying LLM post-training."""

    audits = []
    for row in kpis.to_dict("records"):
        audit = build_reward_audit(row, profile)
        audits.append({
            "run_id": row["run_id"],
            "policy_rung": row["rung"],
            "nr_mode": row["nr_mode"],
            **audit,
            "rollout_trace": {
                "policy_version": simulator_version,
                "environment": "synthetic_aircraft_maintenance_policy_comparison",
                "action": row["rung"],
                "outcome_reference": row["run_id"],
                "serving_status": "schema_only_no_llm_inference",
            },
        })

    best = max(audits, key=lambda item: float(item["reward"])) if audits else None
    return {
        "comparison_id": comparison_id,
        "scope": {
            "label": "Synthetic RL/LLM evaluation design demo",
            "implements": [
                "deterministic KPI-based reward verification",
                "policy rollout observability schema",
                "optional small-scale MoE training artifact when generated separately",
            ],
            "does_not_implement": [
                "LLM RLHF or RLAIF training",
                "LLM inference serving in a rollout loop",
                "multi-node training or MoE expert parallelism",
            ],
        },
        "verifiable_reward_audits": audits,
        "best_verified_rollout": best,
        "llm_as_judge_protocol": {
            "status": "design_only_not_used_for_training",
            "grounding_requirement": "Judge receives only the KPI/profile evidence bundle and must cite run_id fields.",
            "rubric": [
                "Does the explanation preserve the verifiable KPI values?",
                "Does it identify a failed safety or service check?",
                "Does it avoid claims not supported by the evidence?",
            ],
            "calibration_plan": "Compare judge scores with a held-out human-labelled rubric set before using any score as a reward.",
        },
        "inference_rollout_design": {
            "status": "schema_only_no_live_server",
            "required_fields": [
                "policy_version",
                "prompt_or_state_reference",
                "sampled_action_or_trace",
                "log_probability",
                "latency_ms",
                "reward_components",
                "verifier_result",
            ],
            "staleness_guard": "A learner must reject rollout records whose policy_version is outside its configured lag window.",
        },
        "moe_monitoring_design": {
            "status": "populated_from_small_scale_training" if moe_training else "design_only",
            "metrics": [
                "router_entropy",
                "expert_load_fraction",
                "capacity_excess_fraction",
                "per_expert_gradient_norm",
                "all_to_all_communication_time_for_expert_parallel_training",
            ],
            "training_summary": dict(moe_training) if moe_training else None,
        },
    }


def write_evaluation_artifact(
    comparison_dir: Path,
    artifact: Mapping[str, Any],
) -> Path:
    path = comparison_dir / "rl_llm_evaluation.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return path
