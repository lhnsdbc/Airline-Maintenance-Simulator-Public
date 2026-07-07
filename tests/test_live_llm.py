import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from analyst.live_llm import generate_grounded_llm_report


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


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


class LiveLlmTests(unittest.TestCase):
    def test_generate_report_falls_back_without_api_key(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            comparison_id = "default_run_comparison_seed1"
            write_artifacts(root, comparison_id)

            with patch.dict(os.environ, {}, clear=True):
                result = generate_grounded_llm_report(
                    comparison_id,
                    artifact_dir=root / "artifacts",
                    report_dir=root / "reports",
                    prompt_dir=root / "prompts",
                    output_dir=root / "outputs",
                )

        self.assertFalse(result.used_live_provider)
        self.assertEqual(result.provider, "deterministic")
        self.assertIn("Grounded Experiment Analyst Report", result.text)

    def test_generate_report_uses_openai_when_key_is_configured(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            comparison_id = "default_run_comparison_seed1"
            write_artifacts(root, comparison_id)

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "test-model"}, clear=True):
                with patch("analyst.live_llm.httpx.post", return_value=FakeResponse({"output_text": "Grounded live summary."})) as post:
                    result = generate_grounded_llm_report(
                        comparison_id,
                        provider="openai",
                        artifact_dir=root / "artifacts",
                        report_dir=root / "reports",
                        prompt_dir=root / "prompts",
                        output_dir=root / "outputs",
                    )

        self.assertTrue(result.used_live_provider)
        self.assertEqual(result.model, "test-model")
        self.assertEqual(result.text, "Grounded live summary.")
        self.assertEqual(post.call_args.kwargs["json"]["store"], False)

    def test_generate_report_uses_gemini_when_key_is_configured(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            comparison_id = "default_run_comparison_seed1"
            write_artifacts(root, comparison_id)
            payload = {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "Gemini grounded summary."}]
                    }
                }]
            }

            with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "test-gemini"}, clear=True):
                with patch("analyst.live_llm.httpx.post", return_value=FakeResponse(payload)) as post:
                    result = generate_grounded_llm_report(
                        comparison_id,
                        provider="gemini",
                        artifact_dir=root / "artifacts",
                        report_dir=root / "reports",
                        prompt_dir=root / "prompts",
                        output_dir=root / "outputs",
                    )

        self.assertTrue(result.used_live_provider)
        self.assertEqual(result.model, "test-gemini")
        self.assertEqual(result.text, "Gemini grounded summary.")
        self.assertEqual(post.call_args.kwargs["headers"]["x-goog-api-key"], "test-key")


if __name__ == "__main__":
    unittest.main()
