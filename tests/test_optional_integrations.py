import argparse
import importlib.util
import json
import sys
import uuid
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.synthetic_experiment import run_experiment
from retrieval.search import build_documents
from retrieval.vector import build_vector_index, vector_search


CHROMA_AVAILABLE = importlib.util.find_spec("chromadb") is not None
MLFLOW_AVAILABLE = importlib.util.find_spec("mlflow") is not None
REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_AVAILABLE = (REPO_ROOT / "Data" / "input" / "Scenarios_simulation.csv").exists()


class OptionalIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(CHROMA_AVAILABLE, "chromadb is not installed")
    @unittest.skipIf(sys.platform == "win32", "chromadb native add path is unstable in this Windows test environment")
    def test_chroma_backend_indexes_and_searches_evidence(self):
        root = REPO_ROOT / "artifacts" / "test_chroma_optional" / uuid.uuid4().hex
        artifact_dir = root / "artifacts"
        report_dir = root / "reports"
        index_dir = root / "index"
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
        docs = build_documents(artifact_dir, report_dir)

        summary = build_vector_index(
            artifact_dir=artifact_dir,
            report_dir=report_dir,
            index_dir=index_dir,
            backend="chroma",
        )
        results = vector_search(
            "predicted uncovered",
            artifact_dir=artifact_dir,
            report_dir=report_dir,
            index_dir=index_dir,
            backend="chroma",
            nr_mode="predicted",
        )

        self.assertEqual(summary["document_count"], len(docs))
        self.assertTrue(results)
        self.assertEqual(results[0]["metadata"]["run_id"], "run_b")

    @unittest.skipUnless(MLFLOW_AVAILABLE and FIXTURES_AVAILABLE, "mlflow or synthetic fixtures are not available")
    def test_mlflow_manifest_reports_logging_when_mlflow_is_installed(self):
        with TemporaryDirectory() as tmp:
            summary_dir = run_experiment(argparse.Namespace(
                scenario="default_run",
                seed=20260706,
                output_dir=tmp,
            ))
            manifest = json.loads((summary_dir / "mlflow_manifest.json").read_text(encoding="utf-8"))

        self.assertTrue(manifest["mlflow_logged"])
        self.assertEqual(manifest["experiment_name"], "synthetic-policy-comparison")


if __name__ == "__main__":
    unittest.main()
