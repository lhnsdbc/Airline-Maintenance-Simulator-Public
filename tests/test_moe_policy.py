import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.moe_policy import MoeTrainingConfig, _torch_modules, build_synthetic_bandit, train_moe_policy


PROFILE = {
    "fleet_utilization": 0.25,
    "mean_nr_probability": 0.3,
    "policy_task_count": 400,
    "rotation_count": 127,
}


class MoePolicyTests(unittest.TestCase):
    def test_synthetic_bandit_is_seeded_and_has_three_actions(self):
        config = MoeTrainingConfig(seed=123, training_examples=16)
        first_states, first_rewards = build_synthetic_bandit(PROFILE, config)
        second_states, second_rewards = build_synthetic_bandit(PROFILE, config)

        self.assertEqual(first_states.shape, (16, 8))
        self.assertEqual(first_rewards.shape, (16, 3))
        self.assertTrue((first_states == second_states).all())
        self.assertTrue((first_rewards == second_rewards).all())

    def test_training_writes_scope_labelled_artifact_when_torch_is_available(self):
        try:
            _torch_modules()
        except RuntimeError:
            self.skipTest("optional PyTorch dependency is not installed")

        with TemporaryDirectory() as tmp:
            result = train_moe_policy(
                PROFILE,
                Path(tmp),
                MoeTrainingConfig(epochs=1, training_examples=64, batch_size=32),
            )

        self.assertTrue(result["debug_invariants"]["finite_metrics"])
        self.assertEqual(result["distributed"]["world_size"], 1)
        self.assertIn("LLM post-training or RLHF", result["scope"]["does_not_implement"])


if __name__ == "__main__":
    unittest.main()
