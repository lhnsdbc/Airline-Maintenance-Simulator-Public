import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from retrieval.search import build_documents, search_artifacts


class RetrievalTests(unittest.TestCase):
    def test_search_filters_by_metadata(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifacts"
            report_dir = root / "reports"
            comparison_dir = artifact_dir / "default_run_comparison_seed1"
            comparison_dir.mkdir(parents=True)
            report_dir.mkdir(parents=True)
            (comparison_dir / "kpis.csv").write_text(
                "run_id,rung,nr_mode,policy_quality_score,nr_uncovered_labor_hours\n"
                "run_a,R0,static,10,3\n"
                "run_b,R2,predicted,20,0\n",
                encoding="utf-8",
            )
            (comparison_dir / "synthetic_profile.json").write_text(
                json.dumps({"scenario": "default_run", "flight_leg_count": 311}),
                encoding="utf-8",
            )
            (report_dir / "default_run_comparison_seed1_analyst_report.md").write_text(
                "Best run is run_b with predicted NR mode.",
                encoding="utf-8",
            )

            docs = build_documents(artifact_dir, report_dir)
            results = search_artifacts(
                "predicted uncovered",
                artifact_dir=artifact_dir,
                report_dir=report_dir,
                nr_mode="predicted",
            )

        self.assertEqual(len(docs), 4)
        self.assertTrue(results)
        self.assertEqual(results[0]["metadata"]["run_id"], "run_b")


if __name__ == "__main__":
    unittest.main()
