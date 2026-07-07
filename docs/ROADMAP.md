# Public Roadmap

## Phase 0: Public-Safe Baseline

- Keep only synthetic inputs.
- Generate fixtures locally.
- Keep generated data and outputs ignored by Git.
- Document the data boundary clearly.

## Phase 1: Experiment Tracking

- Add local experiment records with optional MLflow mirroring. Done for the synthetic policy-comparison workflow.
- Log scenario ID, seed, simulator version, policy rung, policy mode, KPIs, and artifacts. Done for the synthetic policy-comparison workflow.
- Add comparison-level MLflow visibility. Done with `mlflow_manifest.json` and optional `requirements-mlops.txt`.
- Add a fixed-seed reproducibility check. Done for the synthetic KPI function.
- Next: wire the same recorder into full simulator runs once solver-dependent workflows are stable on the public synthetic profile.

## Phase 2: Policy Dashboard

- Add a dashboard app. Done with Dash using the existing dependency set.
- Show policy KPI tables, distributions, failure cases, and scenario sensitivity. Partly done: polished policy KPI table, primary KPI chart, and NR reserve/realized chart are in place.
- Use local MLflow runs or exported CSV summaries as the data source. Done for exported synthetic tracking CSV summaries.
- Next: add scenario filters and richer failure-case drilldowns once more tracked runs exist.

## Phase 3: API Service

- Add FastAPI endpoints for simulation, policy comparison, experiment lookup, and health checks. Partly done: health, synthetic profile, synthetic policy comparison, and experiment artifact lookup endpoints are in place.
- Validate inputs with Pydantic. Done for synthetic policy comparison requests.
- Return experiment IDs and reproducibility metadata. Done for synthetic comparison artifacts.
- Next: expose full simulator execution once solver-dependent workflows are stable on the public synthetic profile.

## Phase 4: Docker And CI

- Add a Dockerfile for the API service. Done for the public synthetic FastAPI service.
- Add dashboard deployment packaging. Done with `Dockerfile.dashboard`.
- Add public deployment blueprint. Done with `render.yaml` for separate API and dashboard services.
- Add CI for unit tests, synthetic-data generation, smoke simulation, and API health checks. Done for the public synthetic workflow.
- Next: add a live API health-check step in CI after dependency installation and extend smoke coverage to full simulator execution when solver-dependent workflows are stable.

## Phase 5: Grounded LLM Analyst

- Generate short experiment summaries from run metadata and KPI tables. Done as a deterministic grounded analyst report.
- Require citations to scenario IDs, run IDs, metric values, and policy versions. Done for run IDs and metric values; policy versions are represented by policy rungs/modes in the synthetic workflow.
- Keep the LLM layer explanatory, not decision-authoritative. Done in the report caveat.
- Add an LLM-ready prompt package. Done as a provider-neutral JSON export with strict grounding rules and evidence records.
- Add an optional live model adapter that rewrites only the grounded report facts when an API key is available. Done for Gemini, OpenAI, and Anthropic provider keys with deterministic fallback.
- Next: add streaming and provider-specific structured-output validation.

## Phase 6: Retrieval And Monitoring

- Index synthetic experiment summaries and failure cases. Done for KPI rows, synthetic profiles, and grounded analyst reports.
- Add metadata filters for scenario, policy, date, and metric type. Partly done: comparison ID, rung, NR mode, and artifact kind filters are available.
- Add vector retrieval for RAG over evidence artifacts. Done with a dependency-free local vector index and optional Chroma backend using project-owned embeddings.
- Expose lightweight request/runtime metrics and structured logs. Done for in-process API metrics at `/metrics`; structured logs remain a future extension.
- Next: add retrieval evaluation queries and Prometheus-compatible metrics export if needed for deployment-oriented evaluation.
