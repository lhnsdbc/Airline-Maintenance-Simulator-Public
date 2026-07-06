import unittest

import pandas as pd

from analyst.llm_prompt import build_prompt_package


class LlmPromptTests(unittest.TestCase):
    def test_prompt_package_contains_grounding_rules_and_evidence(self):
        kpis = pd.DataFrame([
            {
                "run_id": "run_a",
                "rung": "R2",
                "nr_mode": "predicted",
                "policy_quality_score": 12.3456,
                "flights_delay_dep": 100.0,
                "interval_spillage": 5.0,
                "nr_uncovered_labor_hours": 0.0,
                "completion_factor": 0.99,
            }
        ])
        package = build_prompt_package(
            "comparison_a",
            "Grounded report with run_a.",
            kpis,
            {"flight_leg_count": 311},
        )

        self.assertIn("Use only the evidence JSON", package["system"])
        self.assertEqual(package["evidence"]["kpi_records"][0]["run_id"], "run_a")
        self.assertEqual(package["evidence"]["scenario_profile"]["flight_leg_count"], 311)


if __name__ == "__main__":
    unittest.main()
