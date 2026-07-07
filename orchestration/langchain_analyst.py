"""LangChain orchestration over grounded simulator evidence.

The chain coordinates existing project capabilities: artifact loading, report
generation, retrieval, and prompt packaging. It deliberately keeps model calls
outside the chain so the public workflow stays reproducible without API keys.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from analyst.experiment_report import DEFAULT_ARTIFACT_DIR, DEFAULT_REPORT_DIR, load_artifacts, write_report
from analyst.llm_prompt import DEFAULT_PROMPT_DIR, build_prompt_package
from retrieval.search import search_artifacts


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORCHESTRATION_DIR = REPO_ROOT / "reports" / "orchestration"


def _require_langchain_core():
    try:
        from langchain_core.runnables import RunnableLambda
    except ImportError as exc:
        raise RuntimeError(
            "LangChain orchestration requires langchain-core. "
            "Install optional dependencies with: pip install -r requirements-genai.txt"
        ) from exc
    return RunnableLambda


def _source_summary(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "doc_id": item["doc_id"],
            "source": item["source"],
            "score": item["score"],
            "metadata": item["metadata"],
        }
        for item in results
    ]


def build_maintenance_analyst_chain(
    *,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
    prompt_dir: Path = DEFAULT_PROMPT_DIR,
    retrieval_limit: int = 5,
):
    """Build a LangChain runnable for evidence-grounded maintenance analysis."""

    RunnableLambda = _require_langchain_core()

    def prepare_context(inputs: Dict[str, Any]) -> Dict[str, Any]:
        comparison_id = str(inputs["comparison_id"])
        question = str(inputs.get("question") or "Summarize the maintenance policy comparison.")
        report_path = report_dir / f"{comparison_id}_analyst_report.md"
        if not report_path.exists():
            write_report(comparison_id, artifact_dir, report_dir)
        kpis, profile = load_artifacts(comparison_id, artifact_dir)
        report_markdown = report_path.read_text(encoding="utf-8")
        return {
            "comparison_id": comparison_id,
            "question": question,
            "kpis": kpis,
            "profile": profile,
            "report_markdown": report_markdown,
            "trace": ["loaded_artifacts", "ensured_grounded_report"],
        }

    def retrieve_context(state: Dict[str, Any]) -> Dict[str, Any]:
        results = search_artifacts(
            state["question"],
            artifact_dir=artifact_dir,
            report_dir=report_dir,
            comparison_id=state["comparison_id"],
            limit=retrieval_limit,
        )
        return {
            **state,
            "retrieval_results": results,
            "trace": state["trace"] + ["retrieved_relevant_evidence"],
        }

    def package_response(state: Dict[str, Any]) -> Dict[str, Any]:
        package = build_prompt_package(
            state["comparison_id"],
            state["report_markdown"],
            state["kpis"],
            state["profile"],
        )
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = prompt_dir / f"{state['comparison_id']}_langchain_prompt.json"
        prompt_path.write_text(json.dumps(package, indent=2, sort_keys=True), encoding="utf-8")

        return {
            "framework": "langchain-core",
            "comparison_id": state["comparison_id"],
            "question": state["question"],
            "trace": state["trace"] + ["built_grounded_prompt_package"],
            "retrieved_sources": _source_summary(state["retrieval_results"]),
            "prompt_package_path": str(prompt_path),
            "answer": (
                "Prepared an evidence-grounded prompt package for the requested maintenance "
                "analysis. Use the retrieved_sources list to audit which artifacts grounded "
                "the package before sending it to a live LLM provider."
            ),
        }

    return (
        RunnableLambda(prepare_context)
        | RunnableLambda(retrieve_context)
        | RunnableLambda(package_response)
    )


def run_maintenance_analyst_chain(
    comparison_id: str,
    question: str = "Summarize the maintenance policy comparison.",
    *,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
    prompt_dir: Path = DEFAULT_PROMPT_DIR,
    output_dir: Path = DEFAULT_ORCHESTRATION_DIR,
) -> Dict[str, Any]:
    """Run the LangChain orchestration path and persist a JSON trace."""

    chain = build_maintenance_analyst_chain(
        artifact_dir=artifact_dir,
        report_dir=report_dir,
        prompt_dir=prompt_dir,
    )
    result = chain.invoke({"comparison_id": comparison_id, "question": question})

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{comparison_id}_langchain_trace.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return {**result, "orchestration_trace_path": str(output_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LangChain orchestration over maintenance evidence.")
    parser.add_argument("comparison_id", help="Comparison artifact folder, e.g. default_run_comparison_seed20260706")
    parser.add_argument(
        "--question",
        default="Summarize the maintenance policy comparison.",
        help="Analysis question used for retrieval and prompt packaging.",
    )
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--prompt-dir", default=str(DEFAULT_PROMPT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_ORCHESTRATION_DIR))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_maintenance_analyst_chain(
        args.comparison_id,
        args.question,
        artifact_dir=Path(args.artifact_dir),
        report_dir=Path(args.report_dir),
        prompt_dir=Path(args.prompt_dir),
        output_dir=Path(args.output_dir),
    )
    print(f"Wrote LangChain orchestration trace to {result['orchestration_trace_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

