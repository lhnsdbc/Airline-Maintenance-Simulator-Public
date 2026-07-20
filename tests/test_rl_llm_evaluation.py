import unittest

import pandas as pd

from experiments.rl_llm_evaluation import build_evaluation_artifact, build_reward_audit


PROFILE = {
    "flight_leg_count": 311,
    "aircraft_count": 31,
}

ROW = {
    "run_id": "default_run_R2_predicted_seed123",
    "rung": "R2",
    "nr_mode": "predicted",
    "policy_quality_score": 62.0,
    "flights_delay_dep": 3000.0,
    "interval_spillage": 15.0,
    "nr_uncovered_labor_hours": 0.0,
    "completion_factor": 0.99,
}


class RlLlmEvaluationTests(unittest.TestCase):
    def test_reward_audit_is_deterministic_and_surfaces_checks(self):
        first = build_reward_audit(ROW, PROFILE)
        second = build_reward_audit(ROW, PROFILE)

        self.assertEqual(first, second)
        self.assertTrue(first["all_checks_passed"])
        self.assertGreater(first["reward"], 0.0)

    def test_artifact_keeps_llm_judge_and_moe_scope_separate(self):
        artifact = build_evaluation_artifact(
            "comparison_seed123",
            pd.DataFrame([ROW]),
            PROFILE,
            "abc123",
        )

        self.assertEqual(artifact["llm_as_judge_protocol"]["status"], "design_only_not_used_for_training")
        self.assertEqual(artifact["moe_monitoring_design"]["status"], "design_only")
        self.assertIn("LLM RLHF or RLAIF training", artifact["scope"]["does_not_implement"])
        self.assertEqual(artifact["best_verified_rollout"]["run_id"], ROW["run_id"])


if __name__ == "__main__":
    unittest.main()
