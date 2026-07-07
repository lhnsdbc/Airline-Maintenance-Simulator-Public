"""Optional live LLM generation for grounded experiment reports.

The deterministic analyst report and prompt package remain the source of truth. This
module only rewrites the supplied evidence when a provider API key is configured.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from analyst.experiment_report import DEFAULT_ARTIFACT_DIR, DEFAULT_REPORT_DIR, load_artifacts, write_report
from analyst.llm_prompt import DEFAULT_PROMPT_DIR, build_prompt_package
from experiments.tracking import write_json


DEFAULT_LLM_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "llm_outputs"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


@dataclass(frozen=True)
class LlmGenerationResult:
    provider: str
    model: str
    used_live_provider: bool
    text: str
    output_path: Path
    metadata_path: Path


def _prompt_text(package: Dict[str, Any]) -> str:
    return (
        f"{package['user']}\n\n"
        "Evidence JSON:\n"
        f"{json.dumps(package['evidence'], indent=2, sort_keys=True)}"
    )


def _extract_openai_text(payload: Dict[str, Any]) -> str:
    if payload.get("output_text"):
        return str(payload["output_text"]).strip()

    parts = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if text:
                parts.append(str(text))
    return "\n".join(parts).strip()


def _extract_anthropic_text(payload: Dict[str, Any]) -> str:
    parts = []
    for block in payload.get("content", []):
        if block.get("type") == "text" and block.get("text"):
            parts.append(str(block["text"]))
    return "\n".join(parts).strip()


def _extract_gemini_text(payload: Dict[str, Any]) -> str:
    parts = []
    for candidate in payload.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                parts.append(str(text))
    return "\n".join(parts).strip()


def _call_openai(package: Dict[str, Any], api_key: str, model: str, timeout_seconds: float) -> str:
    response = httpx.post(
        OPENAI_RESPONSES_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": [
                {"role": "system", "content": package["system"]},
                {"role": "user", "content": _prompt_text(package)},
            ],
            "max_output_tokens": 900,
            "store": False,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    text = _extract_openai_text(response.json())
    if not text:
        raise RuntimeError("OpenAI response did not contain text output")
    return text


def _call_anthropic(package: Dict[str, Any], api_key: str, model: str, timeout_seconds: float) -> str:
    response = httpx.post(
        ANTHROPIC_MESSAGES_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 900,
            "system": package["system"],
            "messages": [{"role": "user", "content": _prompt_text(package)}],
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    text = _extract_anthropic_text(response.json())
    if not text:
        raise RuntimeError("Anthropic response did not contain text output")
    return text


def _call_gemini(package: Dict[str, Any], api_key: str, model: str, timeout_seconds: float) -> str:
    response = httpx.post(
        GEMINI_GENERATE_URL.format(model=model),
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "systemInstruction": {
                "parts": [{"text": package["system"]}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": _prompt_text(package)}],
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 900,
                "temperature": 0.2,
            },
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    text = _extract_gemini_text(response.json())
    if not text:
        raise RuntimeError("Gemini response did not contain text output")
    return text


def _select_provider(requested_provider: str) -> str:
    if requested_provider != "auto":
        return requested_provider
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    return "deterministic"


def generate_grounded_llm_report(
    comparison_id: str,
    *,
    provider: str = "auto",
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
    prompt_dir: Path = DEFAULT_PROMPT_DIR,
    output_dir: Path = DEFAULT_LLM_OUTPUT_DIR,
    timeout_seconds: float = 45.0,
) -> LlmGenerationResult:
    report_path = report_dir / f"{comparison_id}_analyst_report.md"
    if not report_path.exists():
        write_report(comparison_id, artifact_dir, report_dir)
    report_markdown = report_path.read_text(encoding="utf-8")
    kpis, profile = load_artifacts(comparison_id, artifact_dir)
    package = build_prompt_package(comparison_id, report_markdown, kpis, profile)

    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompt_dir / f"{comparison_id}_llm_prompt.json"
    prompt_path.write_text(json.dumps(package, indent=2, sort_keys=True), encoding="utf-8")

    selected_provider = _select_provider(provider)
    used_live_provider = False
    model = "deterministic-grounded-report"
    text = report_markdown

    if selected_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            model = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
            text = _call_openai(package, api_key, model, timeout_seconds)
            used_live_provider = True
    elif selected_provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
            text = _call_anthropic(package, api_key, model, timeout_seconds)
            used_live_provider = True
    elif selected_provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
            text = _call_gemini(package, api_key, model, timeout_seconds)
            used_live_provider = True
    elif selected_provider != "deterministic":
        raise ValueError("provider must be one of: auto, deterministic, openai, anthropic, gemini")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{comparison_id}_llm_report.md"
    metadata_path = output_dir / f"{comparison_id}_llm_report_metadata.json"
    output_path.write_text(text, encoding="utf-8")
    write_json(metadata_path, {
        "comparison_id": comparison_id,
        "provider": selected_provider,
        "model": model,
        "used_live_provider": used_live_provider,
        "prompt_package_path": str(prompt_path),
        "output_path": str(output_path),
    })
    return LlmGenerationResult(selected_provider, model, used_live_provider, text, output_path, metadata_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an optional live grounded LLM report.")
    parser.add_argument("comparison_id", help="Comparison artifact folder, e.g. default_run_comparison_seed20260706")
    parser.add_argument("--provider", choices=["auto", "deterministic", "openai", "anthropic", "gemini"], default="auto")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--prompt-dir", default=str(DEFAULT_PROMPT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_LLM_OUTPUT_DIR))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = generate_grounded_llm_report(
        args.comparison_id,
        provider=args.provider,
        artifact_dir=Path(args.artifact_dir),
        report_dir=Path(args.report_dir),
        prompt_dir=Path(args.prompt_dir),
        output_dir=Path(args.output_dir),
    )
    mode = "live" if result.used_live_provider else "deterministic fallback"
    print(f"Wrote {mode} LLM report to {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
