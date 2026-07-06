"""Run a key-free smoke test for the public portfolio workflow."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from retrieval.vector import vector_search

COMPARISON_ID = "default_run_comparison_seed20260706"


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> int:
    python = sys.executable
    run([python, "generate_dummy_data.py"])
    run([python, "generate_mock_nr_artifact.py"])
    run([python, "-m", "experiments.synthetic_experiment", "--scenario", "default_run", "--seed", "20260706"])
    run([python, "-m", "analyst.live_llm", COMPARISON_ID, "--provider", "deterministic"])
    run([python, "-m", "retrieval.vector", "build", "--backend", "local"])

    results = vector_search("predicted uncovered", nr_mode="predicted", limit=3)
    if not results:
        raise SystemExit("Vector retrieval smoke check returned no predicted results")
    if any(result["metadata"].get("nr_mode") != "predicted" for result in results):
        raise SystemExit("Vector retrieval smoke check returned a non-predicted result")

    print("Public workflow smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
