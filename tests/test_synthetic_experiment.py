import unittest

from experiments.synthetic_experiment import compute_metrics


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


if __name__ == "__main__":
    unittest.main()
