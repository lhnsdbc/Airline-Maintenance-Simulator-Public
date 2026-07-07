# Deployment Runbook

This project is packaged as two web services:

- API service: `Dockerfile`, FastAPI on `/health`, `/metrics`, `/search`, `/rag/search`, and experiment endpoints.
- Dashboard service: `Dockerfile.dashboard`, Dash app for KPI charts, grounded analyst text, and LLM prompt packages.

The repository includes `render.yaml` for Render Blueprint deployment. Other container platforms can use the same Dockerfiles.

## Current Public Demo

- Dashboard: https://maintenance-simulator-dashboard.onrender.com
- API: https://maintenance-simulator-api.onrender.com
- API docs: https://maintenance-simulator-api.onrender.com/docs

Render free services may sleep after inactivity. A cold first request can take longer than normal.

## Render Blueprint

1. Push the latest `main` branch to GitHub.
2. In Render, choose **New > Blueprint**.
3. Connect the GitHub repository.
4. Select `render.yaml`.
5. Create the two services:
   - `maintenance-simulator-api`
   - `maintenance-simulator-dashboard`
6. Wait for both Docker builds to finish.
7. Copy the public service URLs into your notes/CV draft.

## Post-Deploy Checks

Replace the URLs below with the Render URLs:

```powershell
Invoke-RestMethod "https://YOUR-API.onrender.com/health"
Invoke-RestMethod "https://YOUR-API.onrender.com/"
Invoke-RestMethod "https://YOUR-API.onrender.com/metrics"
Invoke-RestMethod "https://YOUR-API.onrender.com/experiments"
Invoke-RestMethod "https://YOUR-API.onrender.com/rag/search?q=predicted%20uncovered&nr_mode=predicted"
Invoke-RestMethod -Method Post "https://YOUR-API.onrender.com/experiments/default_run_comparison_seed20260706/llm-report" -ContentType "application/json" -Body '{"provider":"deterministic"}'
Invoke-RestMethod "https://YOUR-DASHBOARD.onrender.com/health"
```

Or run the bundled verifier:

```powershell
py scripts/verify_live_demo.py --api-url "https://YOUR-API.onrender.com" --dashboard-url "https://YOUR-DASHBOARD.onrender.com"
```

The same check can be run from GitHub Actions with the manual `verify-live-demo` workflow.

Then open the dashboard URL in a browser and confirm:

- The selected experiment is `default_run_comparison_seed20260706`.
- KPI cards and charts are visible.
- The Grounded Analyst panel contains the generated report.
- The LLM Prompt Package panel contains evidence JSON.

## Optional Provider Keys

The public demo does not need paid LLM keys. If you want live provider-backed reports, add one of these service environment variables to the API service:

- `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- `GEMINI_MODEL`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `ANTHROPIC_API_KEY`
- `ANTHROPIC_MODEL`

The deterministic fallback remains available with:

```json
{"provider": "deterministic"}
```

Use live providers with:

```json
{"provider": "gemini"}
```

or:

```json
{"provider": "openai"}
```

or:

```json
{"provider": "anthropic"}
```

## Honest CV Wording

Before deployment, use:

> Packaged a FastAPI/Dash simulation analytics workflow for cloud deployment with Docker, CI, health checks, runtime metrics, RAG search, and grounded LLM reporting.

After the public services are live, use:

> Deployed a containerized FastAPI/Dash aircraft maintenance analytics demo with experiment tracking, RAG over simulation artifacts, runtime metrics, and evidence-bounded LLM reporting.

Do not claim production operations, high availability, Kubernetes, or real airline data. This is a public synthetic-data deployment.
