import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from dashboard.app import _add_marker_size, _build_llm_panel_content, create_app, load_comparisons


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

    def test_health_endpoint_reports_artifact_rows(self):
        with TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            run_dir = artifact_dir / "default_run_comparison_seed1"
            run_dir.mkdir(parents=True)
            pd.DataFrame([{
                "run_id": "demo",
                "rung": "R0",
                "nr_mode": "static",
                "policy_quality_score": 1.0,
            }]).to_csv(run_dir / "kpis.csv", index=False)
            app = create_app(artifact_dir)

            response = app.server.test_client().get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "ok")
        self.assertEqual(response.json["comparison_count"], 1)
        self.assertEqual(response.json["row_count"], 1)

    def test_marker_size_is_positive_when_quality_score_is_negative(self):
        df = pd.DataFrame({"policy_quality_score": [-5.998, 12.557]})

        sized = _add_marker_size(df)

        self.assertTrue((sized["marker_size_score"] > 0).all())

    def test_llm_panel_builds_grounded_report_and_prompt_package(self):
        with TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            comparison_id = "default_run_comparison_seed1"
            comparison_dir = artifact_dir / comparison_id
            comparison_dir.mkdir(parents=True)
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
            kpis = pd.DataFrame([{
                "run_id": "run_a",
                "rung": "R2",
                "nr_mode": "predicted",
                "policy_quality_score": 32.084,
                "flights_delay_dep": 3528.14,
                "interval_spillage": 17.82,
                "nr_uncovered_labor_hours": 0.0,
                "completion_factor": 0.9874,
            }])

            report, package_json = _build_llm_panel_content(comparison_id, kpis, artifact_dir)

        package = json.loads(package_json)
        self.assertIn("Grounded Experiment Analyst Report", report)
        self.assertIn("Use only the evidence JSON", package["system"])
        self.assertEqual(package["evidence"]["kpi_records"][0]["run_id"], "run_a")


if __name__ == "__main__":
    unittest.main()
