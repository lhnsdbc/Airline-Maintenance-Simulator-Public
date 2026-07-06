import argparse
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.synthetic_experiment import compute_metrics, run_experiment


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_AVAILABLE = (REPO_ROOT / "Data" / "input" / "Scenarios_simulation.csv").exists()


class SyntheticExperimentTests(unittest.TestCase):
    def test_metrics_are_deterministic_for_same_inputs(self):
        profile = {
            "flight_leg_count": 311,
            "rotation_count": 127,
            "fleet_utilization": 0.25,
            "total_task_labor_hours": 900.0,
            "mean_nr_q50_hours": 2.4,
            "mean_nr_probability": 0.3,
            "hangar_a_slot_templates": 3,
        }
        first = compute_metrics(profile, "R1", "predicted", 123)
        second = compute_metrics(profile, "R1", "predicted", 123)
        self.assertEqual(first, second)

    def test_predicted_mode_reduces_uncovered_nr_for_same_rung(self):
        profile = {
            "flight_leg_count": 311,
            "rotation_count": 127,
            "fleet_utilization": 0.25,
            "total_task_labor_hours": 900.0,
            "mean_nr_q50_hours": 2.4,
            "mean_nr_probability": 0.3,
            "hangar_a_slot_templates": 3,
        }
        static = compute_metrics(profile, "R0", "static", 123)
        predicted = compute_metrics(profile, "R0", "predicted", 123)
        self.assertLessEqual(
            predicted["nr_uncovered_labor_hours"],
            static["nr_uncovered_labor_hours"],
        )

    @unittest.skipUnless(FIXTURES_AVAILABLE, "synthetic fixtures are not generated")
    def test_run_experiment_writes_mlflow_manifest(self):
        with TemporaryDirectory() as tmp:
            summary_dir = run_experiment(argparse.Namespace(
                scenario="default_run",
                seed=20260706,
                output_dir=tmp,
            ))
            manifest = json.loads((summary_dir / "mlflow_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["experiment_name"], "synthetic-policy-comparison")
        self.assertEqual(manifest["params"]["tracking_scope"], "comparison_summary")
        self.assertEqual(manifest["metrics"]["run_count"], 6)


if __name__ == "__main__":
    unittest.main()
