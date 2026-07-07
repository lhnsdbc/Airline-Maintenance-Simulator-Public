import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from orchestration.langchain_analyst import run_maintenance_analyst_chain


LANGCHAIN_CORE_AVAILABLE = importlib.util.find_spec("langchain_core") is not None


def write_artifacts(root: Path, comparison_id: str) -> None:
    comparison_dir = root / "artifacts" / comparison_id
    comparison_dir.mkdir(parents=True)
    pd.DataFrame([{
        "run_id": "run_a",
        "rung": "R2",
        "nr_mode": "predicted",
        "policy_quality_score": 32.084,
        "flights_delay_dep": 3528.14,
        "interval_spillage": 17.82,
        "nr_uncovered_labor_hours": 0.0,
        "completion_factor": 0.9874,
    }]).to_csv(comparison_dir / "kpis.csv", index=False)
    (comparison_dir / "synthetic_profile.json").write_text(
        json.dumps({
            "aircraft_count": 31,
            "airport_count": 38,
            "rotation_count": 127,
            "flight_leg_count": 311,
            "duration_days": 7,
        }),
        encoding="utf-8",
    )


class LangChainOrchestrationTests(unittest.TestCase):
    @unittest.skipUnless(LANGCHAIN_CORE_AVAILABLE, "langchain-core is not installed")
    def test_chain_builds_grounded_prompt_and_trace(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            comparison_id = "default_run_comparison_seed1"
            write_artifacts(root, comparison_id)

            result = run_maintenance_analyst_chain(
                comparison_id,
                "predicted uncovered maintenance workload",
                artifact_dir=root / "artifacts",
                report_dir=root / "reports",
                prompt_dir=root / "prompts",
                output_dir=root / "orchestration",
            )

            self.assertTrue(Path(result["prompt_package_path"]).exists())
            self.assertTrue(Path(result["orchestration_trace_path"]).exists())

        self.assertEqual(result["framework"], "langchain-core")
        self.assertIn("retrieved_relevant_evidence", result["trace"])
        self.assertTrue(result["retrieved_sources"])


if __name__ == "__main__":
    unittest.main()
