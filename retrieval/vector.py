"""Vector retrieval over generated experiment evidence.

The default local backend is dependency-free and deterministic for CI. Install
`requirements-rag.txt` and pass `backend="chroma"` to persist the same evidence in
Chroma for a fuller RAG stack.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from retrieval.search import DEFAULT_ARTIFACT_DIR, DEFAULT_REPORT_DIR, SearchDocument, build_documents


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VECTOR_DIR = REPO_ROOT / "artifacts" / "vector_index"
VECTOR_DIM = 128


def _tokens(text: str) -> List[str]:
    return [part.lower() for part in text.replace("_", " ").split() if part.strip()]


def _embed(text: str, dim: int = VECTOR_DIM) -> List[float]:
    vector = [0.0] * dim
    for token in _tokens(text):
        bucket = hash(token) % dim
        vector[bucket] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _cosine(left: Iterable[float], right: Iterable[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _passes_filters(
    metadata: Dict[str, str],
    *,
    comparison_id: Optional[str] = None,
    rung: Optional[str] = None,
    nr_mode: Optional[str] = None,
    kind: Optional[str] = None,
) -> bool:
    return not (
        (comparison_id and metadata.get("comparison_id") != comparison_id)
        or (rung and metadata.get("rung") != rung)
        or (nr_mode and metadata.get("nr_mode") != nr_mode)
        or (kind and metadata.get("kind") != kind)
    )


def _local_index_path(index_dir: Path) -> Path:
    return index_dir / "local_vector_index.json"


def build_local_vector_index(
    documents: Iterable[SearchDocument],
    index_dir: Path = DEFAULT_VECTOR_DIR,
) -> Path:
    index_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for doc in documents:
        rows.append({
            "doc_id": doc.doc_id,
            "source": doc.source,
            "text": doc.text,
            "metadata": doc.metadata,
            "embedding": _embed(doc.text),
        })
    path = _local_index_path(index_dir)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_chroma_vector_index(
    documents: Iterable[SearchDocument],
    index_dir: Path = DEFAULT_VECTOR_DIR,
    collection_name: str = "maintenance_evidence",
) -> int:
    try:
        import chromadb  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install `requirements-rag.txt` to use the Chroma backend.") from exc

    index_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name)
    docs = list(documents)
    if not docs:
        return 0
    collection.add(
        ids=[doc.doc_id for doc in docs],
        documents=[doc.text for doc in docs],
        metadatas=[{**doc.metadata, "source": doc.source} for doc in docs],
        embeddings=[_embed(doc.text) for doc in docs],
    )
    return len(docs)


def build_vector_index(
    *,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
    index_dir: Path = DEFAULT_VECTOR_DIR,
    backend: str = "local",
) -> Dict[str, object]:
    docs = build_documents(artifact_dir, report_dir)
    if backend == "local":
        path = build_local_vector_index(docs, index_dir)
        return {"backend": backend, "document_count": len(docs), "index_path": str(path)}
    if backend == "chroma":
        count = build_chroma_vector_index(docs, index_dir)
        return {"backend": backend, "document_count": count, "index_path": str(index_dir / "chroma")}
    raise ValueError("backend must be one of: local, chroma")


def local_vector_search(
    query: str,
    *,
    index_dir: Path = DEFAULT_VECTOR_DIR,
    limit: int = 5,
    comparison_id: Optional[str] = None,
    rung: Optional[str] = None,
    nr_mode: Optional[str] = None,
    kind: Optional[str] = None,
) -> List[Dict[str, object]]:
    path = _local_index_path(index_dir)
    rows = json.loads(path.read_text(encoding="utf-8"))
    query_embedding = _embed(query)
    results = []
    for row in rows:
        if not _passes_filters(row["metadata"], comparison_id=comparison_id, rung=rung, nr_mode=nr_mode, kind=kind):
            continue
        query_terms = set(_tokens(query))
        row_terms = set(_tokens(row["text"]))
        overlap = len(query_terms & row_terms) / max(1, len(query_terms))
        score = _cosine(query_embedding, row["embedding"]) + overlap
        if score <= 0:
            continue
        results.append({
            "doc_id": row["doc_id"],
            "score": round(score, 4),
            "source": row["source"],
            "metadata": row["metadata"],
            "snippet": row["text"][:260].replace("\n", " "),
        })
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:limit]


def chroma_vector_search(
    query: str,
    *,
    index_dir: Path = DEFAULT_VECTOR_DIR,
    collection_name: str = "maintenance_evidence",
    limit: int = 5,
    comparison_id: Optional[str] = None,
    rung: Optional[str] = None,
    nr_mode: Optional[str] = None,
    kind: Optional[str] = None,
) -> List[Dict[str, object]]:
    try:
        import chromadb  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install `requirements-rag.txt` to use the Chroma backend.") from exc

    where = {
        key: value
        for key, value in {
            "comparison_id": comparison_id,
            "rung": rung,
            "nr_mode": nr_mode,
            "kind": kind,
        }.items()
        if value
    } or None
    collection = chromadb.PersistentClient(path=str(index_dir / "chroma")).get_collection(collection_name)
    response = collection.query(query_embeddings=[_embed(query)], n_results=limit, where=where)
    ids = response.get("ids", [[]])[0]
    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]
    results = []
    for doc_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        results.append({
            "doc_id": doc_id,
            "score": round(1.0 / (1.0 + float(distance)), 4),
            "source": metadata.get("source", ""),
            "metadata": metadata,
            "snippet": document[:260].replace("\n", " "),
        })
    return results


def vector_search(
    query: str,
    *,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
    index_dir: Path = DEFAULT_VECTOR_DIR,
    backend: str = "local",
    limit: int = 5,
    comparison_id: Optional[str] = None,
    rung: Optional[str] = None,
    nr_mode: Optional[str] = None,
    kind: Optional[str] = None,
) -> List[Dict[str, object]]:
    if backend == "local":
        if not _local_index_path(index_dir).exists():
            build_vector_index(artifact_dir=artifact_dir, report_dir=report_dir, index_dir=index_dir, backend=backend)
        return local_vector_search(
            query,
            index_dir=index_dir,
            limit=limit,
            comparison_id=comparison_id,
            rung=rung,
            nr_mode=nr_mode,
            kind=kind,
        )
    if backend == "chroma":
        return chroma_vector_search(
            query,
            index_dir=index_dir,
            limit=limit,
            comparison_id=comparison_id,
            rung=rung,
            nr_mode=nr_mode,
            kind=kind,
        )
    raise ValueError("backend must be one of: local, chroma")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build/query a vector index over experiment evidence.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--backend", choices=["local", "chroma"], default="local")
    build.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    build.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    build.add_argument("--index-dir", default=str(DEFAULT_VECTOR_DIR))
    search = subparsers.add_parser("search")
    search.add_argument("query")
    search.add_argument("--backend", choices=["local", "chroma"], default="local")
    search.add_argument("--index-dir", default=str(DEFAULT_VECTOR_DIR))
    search.add_argument("--limit", type=int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "build":
        print(json.dumps(build_vector_index(
            artifact_dir=Path(args.artifact_dir),
            report_dir=Path(args.report_dir),
            index_dir=Path(args.index_dir),
            backend=args.backend,
        ), indent=2, sort_keys=True))
    else:
        print(json.dumps(vector_search(
            args.query,
            index_dir=Path(args.index_dir),
            backend=args.backend,
            limit=args.limit,
        ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
