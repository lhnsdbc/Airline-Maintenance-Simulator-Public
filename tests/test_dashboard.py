import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from dashboard.app import load_comparisons


class DashboardTests(unittest.TestCase):
    def test_load_comparisons_reads_seeded_kpi_exports(self):
        with TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            run_dir = artifact_dir / "default_run_comparison_seed1"
            run_dir.mkdir(parents=True)
            pd.DataFrame([
                {
                    "run_id": "demo",
                    "rung": "R0",
                    "nr_mode": "static",
                    "policy_quality_score": 1.0,
                }
            ]).to_csv(run_dir / "kpis.csv", index=False)

            df = load_comparisons(artifact_dir)

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["comparison_id"], "default_run_comparison_seed1")


if __name__ == "__main__":
    unittest.main()
