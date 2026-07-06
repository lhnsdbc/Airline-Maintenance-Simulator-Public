# Project Positioning

This repository is a public synthetic-data implementation of an aircraft maintenance simulation and policy-evaluation workflow.

## Scope

The project focuses on reproducible synthetic fixtures, policy comparison, experiment tracking, service interfaces, dashboarding, grounded reporting, retrieval, and lightweight monitoring.

It intentionally does not publish private operational data, derived real-input bundles, model weights, confidential reports, or private repository history.

## Architecture Summary

- Synthetic input generators create aircraft, airport, schedule, maintenance policy, and non-routine workload artifacts.
- Experiment tracking records deterministic policy comparisons across maintenance rungs and non-routine workload modes.
- The dashboard visualizes KPI differences across policy rungs and reserve/realized non-routine workload alignment.
- The FastAPI service exposes health, profile, experiment lookup, policy comparison, search, and runtime metrics endpoints.
- The analyst layer creates deterministic reports and LLM-ready prompt packages from grounded KPI evidence.
- Retrieval indexes local KPI/profile/report artifacts with metadata filters.
- Docker and CI package and verify the public synthetic workflow.

## Boundaries

- Synthetic KPI values are workflow examples, not operational findings.
- The public service is a reproducible local/API workflow, not a live production deployment.
- The project does not claim high-traffic operations, Kubernetes ownership, or production incident response.
- Any optional LLM use must preserve the evidence boundary: outputs should only restate facts present in the generated evidence package.

## Suggested Technical Summary

Public synthetic aircraft maintenance simulation and policy-evaluation platform with reproducible experiment tracking, dashboarding, FastAPI endpoints, Docker/CI packaging, grounded KPI reporting, local retrieval, and runtime monitoring.
