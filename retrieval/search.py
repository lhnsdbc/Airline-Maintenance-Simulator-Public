"""Dependency-free lexical search over generated experiment artifacts."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "experiments"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports"
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class SearchDocument:
    doc_id: str
    source: str
    text: str
    metadata: Dict[str, str]


def _tokens(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _comparison_id_from_path(path: Path) -> str:
    return path.parent.name


def _read_csv_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def build_documents(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> List[SearchDocument]:
    docs: List[SearchDocument] = []

    for kpis_path in sorted(artifact_dir.glob("*_comparison_seed*/kpis.csv")):
        comparison_id = _comparison_id_from_path(kpis_path)
        for row in _read_csv_rows(kpis_path):
            run_id = row.get("run_id", "unknown_run")
            text = " ".join(f"{key} {value}" for key, value in row.items())
            docs.append(SearchDocument(
                doc_id=f"{comparison_id}:{run_id}:kpi",
                source=str(kpis_path),
                text=text,
                metadata={
                    "comparison_id": comparison_id,
                    "run_id": run_id,
                    "rung": row.get("rung", ""),
                    "nr_mode": row.get("nr_mode", ""),
                    "kind": "kpi",
                },
            ))

    for profile_path in sorted(artifact_dir.glob("*_comparison_seed*/synthetic_profile.json")):
        comparison_id = _comparison_id_from_path(profile_path)
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        text = " ".join(f"{key} {value}" for key, value in payload.items())
        docs.append(SearchDocument(
            doc_id=f"{comparison_id}:profile",
            source=str(profile_path),
            text=text,
            metadata={
                "comparison_id": comparison_id,
                "run_id": "",
                "rung": "",
                "nr_mode": "",
                "kind": "profile",
            },
        ))

    for report_path in sorted(report_dir.glob("*_analyst_report.md")):
        comparison_id = report_path.name.replace("_analyst_report.md", "")
        docs.append(SearchDocument(
            doc_id=f"{comparison_id}:report",
            source=str(report_path),
            text=report_path.read_text(encoding="utf-8"),
            metadata={
                "comparison_id": comparison_id,
                "run_id": "",
                "rung": "",
                "nr_mode": "",
                "kind": "report",
            },
        ))

    return docs


def search_documents(
    query: str,
    documents: Iterable[SearchDocument],
    *,
    comparison_id: Optional[str] = None,
    rung: Optional[str] = None,
    nr_mode: Optional[str] = None,
    kind: Optional[str] = None,
    limit: int = 5,
) -> List[Dict[str, object]]:
    query_terms = _tokens(query)
    if not query_terms:
        return []

    filtered = []
    for doc in documents:
        if comparison_id and doc.metadata.get("comparison_id") != comparison_id:
            continue
        if rung and doc.metadata.get("rung") != rung:
            continue
        if nr_mode and doc.metadata.get("nr_mode") != nr_mode:
            continue
        if kind and doc.metadata.get("kind") != kind:
            continue
        filtered.append(doc)

    doc_tokens = {doc.doc_id: _tokens(doc.text) for doc in filtered}
    n_docs = max(1, len(filtered))
    doc_freq: Dict[str, int] = {}
    for terms in doc_tokens.values():
        for term in set(terms):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    results = []
    for doc in filtered:
        terms = doc_tokens[doc.doc_id]
        if not terms:
            continue
        term_counts: Dict[str, int] = {}
        for term in terms:
            term_counts[term] = term_counts.get(term, 0) + 1
        score = 0.0
        for term in query_terms:
            tf = term_counts.get(term, 0)
            if tf == 0:
                continue
            idf = math.log((1 + n_docs) / (1 + doc_freq.get(term, 0))) + 1.0
            score += (1.0 + math.log(tf)) * idf
        if score <= 0:
            continue
        snippet = doc.text[:260].replace("\n", " ")
        results.append({
            "doc_id": doc.doc_id,
            "score": round(score, 4),
            "source": doc.source,
            "metadata": doc.metadata,
            "snippet": snippet,
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:limit]


def search_artifacts(
    query: str,
    *,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
    comparison_id: Optional[str] = None,
    rung: Optional[str] = None,
    nr_mode: Optional[str] = None,
    kind: Optional[str] = None,
    limit: int = 5,
) -> List[Dict[str, object]]:
    return search_documents(
        query,
        build_documents(artifact_dir, report_dir),
        comparison_id=comparison_id,
        rung=rung,
        nr_mode=nr_mode,
        kind=kind,
        limit=limit,
    )
