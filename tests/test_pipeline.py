import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pipeline.run import run_pipeline


class PipelineTests(unittest.TestCase):
    def test_pipeline_writes_medallion_layers_and_status(self):
        repo_root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            status = run_pipeline(repo_root / "Data", repo_root / "artifacts" / "experiments", root)

            self.assertTrue(status["quality_passed"])
            self.assertTrue((root / "bronze" / "aircraft").exists())
            self.assertTrue((root / "silver" / "schedule").exists())
            self.assertTrue((root / "gold" / "operations_by_weekday").exists())
            saved = json.loads((root / "gold" / "pipeline_status" / "latest.json").read_text(encoding="utf-8"))

        self.assertEqual(saved["run_id"], status["run_id"])
        self.assertEqual(saved["status"], "succeeded")


if __name__ == "__main__":
    unittest.main()
