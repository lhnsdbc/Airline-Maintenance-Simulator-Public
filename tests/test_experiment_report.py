import unittest

import pandas as pd

from analyst.experiment_report import build_report


class ExperimentReportTests(unittest.TestCase):
    def test_report_cites_run_ids_and_metrics(self):
        kpis = pd.DataFrame([
            {
                "run_id": "run_static",
                "rung": "R0",
                "nr_mode": "static",
                "policy_quality_score": 10.0,
                "flights_delay_dep": 50.0,
                "interval_spillage": 20.0,
                "nr_uncovered_labor_hours": 3.0,
                "completion_factor": 0.9,
            },
            {
                "run_id": "run_predicted",
                "rung": "R2",
                "nr_mode": "predicted",
                "policy_quality_score": 20.0,
                "flights_delay_dep": 40.0,
                "interval_spillage": 12.0,
                "nr_uncovered_labor_hours": 0.0,
                "completion_factor": 0.95,
            },
        ])
        profile = {
            "aircraft_count": 31,
            "airport_count": 38,
            "rotation_count": 127,
            "flight_leg_count": 311,
            "duration_days": 7,
        }

        report = build_report("demo", kpis, profile)

        self.assertIn("run_predicted", report)
        self.assertIn("20.00", report)
        self.assertIn("311 flight legs", report)


if __name__ == "__main__":
    unittest.main()
