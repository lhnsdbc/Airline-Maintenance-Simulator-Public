# Public Roadmap

## Phase 0: Public-Safe Baseline

- Keep only synthetic inputs.
- Generate fixtures locally.
- Keep generated data and outputs ignored by Git.
- Document the data boundary clearly.

## Phase 1: Experiment Tracking

- Add MLflow local tracking.
- Log scenario ID, seed, simulator version, policy rung, policy mode, KPIs, and artifacts.
- Add a fixed-seed reproducibility check.

## Phase 2: Policy Dashboard

- Add a Streamlit app.
- Show policy KPI tables, distributions, failure cases, and scenario sensitivity.
- Use local MLflow runs or exported CSV summaries as the data source.

## Phase 3: API Service

- Add FastAPI endpoints for simulation, policy comparison, experiment lookup, and health checks.
- Validate inputs with Pydantic.
- Return experiment IDs and reproducibility metadata.

## Phase 4: Docker And CI

- Add a Dockerfile for the API service.
- Add CI for unit tests, synthetic-data generation, smoke simulation, and API health checks.

## Phase 5: Grounded LLM Analyst

- Generate short experiment summaries from run metadata and KPI tables.
- Require citations to scenario IDs, run IDs, metric values, and policy versions.
- Keep the LLM layer explanatory, not decision-authoritative.

## Phase 6: Retrieval And Monitoring

- Index synthetic experiment summaries and failure cases.
- Add metadata filters for scenario, policy, date, and metric type.
- Expose lightweight request/runtime metrics and structured logs.
