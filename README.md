# Aircraft Maintenance ML Simulator

Public, synthetic-data version of an aircraft maintenance simulation project. The goal is to demonstrate production-style simulation, policy evaluation, and ML decision-support workflows without publishing any private airline data.

## What This Project Shows

- Discrete-event simulation for aircraft operations and maintenance planning.
- Baseline and learned maintenance-policy comparison hooks.
- Synthetic input generation for reproducible local runs.
- Experiment runner structure for fixed scenarios, seeds, and policy rungs.
- A roadmap toward experiment tracking, dashboards, API packaging, CI, and LLM-assisted analysis.

## Data Policy

This repository is designed to contain synthetic fixtures only. It intentionally excludes private airline extracts, derived real-input bundles, generated output folders, pickles, trained model weights, and historical Git history from the private working project.

Generate local synthetic inputs with:

```powershell
py generate_dummy_data.py
py generate_mock_nr_artifact.py
```

The generated files are ignored by Git.

## Quick Start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
py generate_dummy_data.py
py generate_mock_nr_artifact.py
py main.py
```

The synthetic generator creates a medium-size one-week scenario with synthetic aircraft, airports, rotations, maintenance slots, maintenance-policy rows, and NR prediction artifacts. Some optimization paths require a solver supported by Pyomo. If a commercial solver is unavailable, use the synthetic smoke workflow first and keep solver-dependent experiments scoped accordingly.

The default generated profile is intentionally large enough to exercise realistic code paths:

- 31 synthetic aircraft registrations.
- 38 synthetic airports.
- 127 rotations and 311 flight legs in a one-week schedule.
- 10 maintenance-slot templates.
- 403 synthetic maintenance-policy rows.
- 404 conditional NR prediction rows including the fleet fallback.

## Experiment Tracking Demo

Run a deterministic policy-comparison experiment over the generated synthetic profile:

```powershell
py -m experiments.synthetic_experiment --scenario default_run --seed 20260706
```

This writes one reproducible local record per policy rung and NR mode under `artifacts/experiments/`, plus a comparison table and short Markdown summary. Each run record includes scenario ID, seed, simulator revision, policy rung, NR mode, metadata, and KPI proxies.

If `mlflow` is installed, the same run params, metrics, and JSON artifacts are also mirrored to a local MLflow experiment named `synthetic-policy-comparison`. MLflow is optional so the synthetic public workflow remains easy to run.

## Dashboard

After creating experiment artifacts, open the policy-comparison dashboard:

```powershell
py -m dashboard.app --port 8050
```

Then visit `http://127.0.0.1:8050`. The dashboard reads `artifacts/experiments/*/kpis.csv`, compares policy rungs and NR modes, and highlights delay, uncovered NR labour, interval spillage, and overall proxy quality.

## API Service

Run the FastAPI service after installing dependencies and generating synthetic fixtures:

```powershell
py -m api.app
```

Key endpoints:

- `GET /health`
- `GET /profile/default_run`
- `POST /compare-policies`
- `GET /experiments`
- `GET /experiments/{comparison_id}/profile`
- `GET /experiments/{comparison_id}/kpis`

The API validates request inputs with Pydantic and returns experiment IDs, reproducibility metadata, and KPI records from the synthetic tracking artifacts.

## Docker And CI

The public service has a lightweight dependency set in `requirements-service.txt` for the synthetic API/dashboard workflow. Build and run the API container with:

```powershell
docker build -t aircraft-maintenance-ml-simulator .
docker run --rm -p 8000:8000 aircraft-maintenance-ml-simulator
```

The image generates synthetic fixtures and a deterministic comparison artifact during build, then serves the API on port `8000`.

GitHub Actions CI is configured to install the public workflow dependencies, generate synthetic fixtures, run the tracked experiment, execute tests, and scan for private-source terms.

## Roadmap

1. Extend the local/optional-MLflow experiment tracker to full simulator runs.
2. Extend the Dash policy-comparison dashboard with scenario filters and historical run comparisons.
3. Extend the FastAPI service from synthetic policy comparison to full simulator workflows.
4. Extend Docker/CI from synthetic service smoke tests to full simulator smoke tests.
5. Add grounded LLM experiment summaries over run metadata and KPI tables.
6. Add retrieval over synthetic experiment logs and reports.
7. Add lightweight service and policy-quality monitoring.

## Portfolio Positioning

This repo is intended to support a CV claim like:

> Built a public aircraft maintenance simulation platform using synthetic operational data, with reproducible experiment tracking, policy comparison workflows, API-ready simulation structure, and a roadmap toward production-style ML evaluation.

Avoid claiming live production deployment, ownership of confidential airline data, or high-traffic service operation from this public version.
