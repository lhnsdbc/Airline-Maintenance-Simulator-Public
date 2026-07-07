# Project Positioning

This repository is a public synthetic-data implementation of an aircraft maintenance simulation and policy-evaluation workflow.

## Scope

The project focuses on reproducible synthetic fixtures, policy comparison, experiment tracking, service interfaces, dashboarding, grounded reporting, retrieval, GenAI orchestration, and lightweight monitoring.

It intentionally does not publish private operational data, derived real-input bundles, model weights, confidential reports, or private repository history.

## Architecture Summary

- Synthetic input generators create aircraft, airport, schedule, maintenance policy, and non-routine workload artifacts.
- Experiment tracking records deterministic policy comparisons across maintenance rungs and non-routine workload modes.
- The dashboard visualizes KPI differences across policy rungs, reserve/realized non-routine workload alignment, grounded analyst reports, and LLM prompt packages.
- The FastAPI service exposes health, profile, experiment lookup, policy comparison, lexical search, vector RAG search, grounded LLM report, and runtime metrics endpoints.
- The analyst layer creates deterministic reports, LLM-ready prompt packages, and optional Gemini/OpenAI/Anthropic-backed summaries from grounded KPI evidence.
- Retrieval indexes local KPI/profile/report artifacts with metadata filters through lexical search, a local vector index, and an optional Chroma backend.
- The optional LangChain orchestration layer coordinates retrieval, grounded reports, and prompt packages while keeping live model calls provider-specific and auditable.
- Docker, Render configuration, and CI package and verify the public synthetic workflow.

## Boundaries

- Synthetic KPI values are workflow examples, not operational findings.
- The public service is deployable as API and dashboard web services, but it is not presented as a high-traffic production system.
- The project does not claim high-traffic operations, Kubernetes ownership, or production incident response.
- Any optional LLM use must preserve the evidence boundary: outputs should only restate facts present in the generated evidence package.

## Suggested Technical Summary

Public synthetic aircraft maintenance simulation and policy-evaluation platform with reproducible experiment tracking, MLflow-ready artifacts, dashboarding, FastAPI endpoints, Docker/Render packaging, RAG over generated evidence, LangChain orchestration, grounded LLM reporting, and runtime monitoring.
