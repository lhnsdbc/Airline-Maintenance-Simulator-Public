import unittest

from experiments.evaluation import EVALUATION_SCENARIOS, apply_scenario, evaluate, summarize


PROFILE = {
    "flight_leg_count": 311,
    "rotation_count": 127,
    "fleet_utilization": 0.25,
    "total_task_labor_hours": 900.0,
    "mean_nr_q50_hours": 2.4,
    "mean_nr_probability": 0.3,
    "hangar_a_slot_templates": 3,
}


class EvaluationTests(unittest.TestCase):
    def test_hangar_profile_reduces_available_a_check_slots(self):
        metrics = {"flights_delay_dep": 100.0, "interval_spillage": 10.0, "a_check_slots_executed": 3.0,
                   "nr_uncovered_labor_hours": 2.0}
        adjusted = apply_scenario(metrics, "constrained_hangar")

        self.assertLess(adjusted["a_check_slots_executed"], metrics["a_check_slots_executed"])
        self.assertGreater(adjusted["interval_spillage"], metrics["interval_spillage"])

    def test_evaluation_has_each_policy_for_every_seed_and_scenario(self):
        rows = evaluate(PROFILE, seeds=[101, 102])

        self.assertEqual(len(rows), len(EVALUATION_SCENARIOS) * 2 * 2)
        self.assertEqual({row["nr_mode"] for row in rows}, {"predicted"})
        self.assertEqual(len(summarize(rows)), len(EVALUATION_SCENARIOS) * 2 * 4)


if __name__ == "__main__":
    unittest.main()
