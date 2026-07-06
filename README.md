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
python generate_dummy_data.py
python generate_mock_nr_artifact.py
```

The generated files are ignored by Git.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python generate_dummy_data.py
python generate_mock_nr_artifact.py
python main.py
```

Some optimization paths require a solver supported by Pyomo. If a commercial solver is unavailable, use the lightweight mock-data workflow first and keep solver-dependent experiments scoped accordingly.

## Roadmap

1. Add MLflow experiment tracking for simulator runs.
2. Build a Streamlit dashboard for policy comparison across scenarios.
3. Wrap simulator workflows with FastAPI endpoints.
4. Add Docker packaging and CI smoke tests.
5. Add grounded LLM experiment summaries over run metadata and KPI tables.
6. Add retrieval over synthetic experiment logs and reports.
7. Add lightweight service and policy-quality monitoring.

## Portfolio Positioning

This repo is intended to support a CV claim like:

> Built a public aircraft maintenance simulation platform using synthetic operational data, with reproducible experiment tracking, policy comparison workflows, API-ready simulation structure, and a roadmap toward production-style ML evaluation.

Avoid claiming live production deployment, ownership of confidential airline data, or high-traffic service operation from this public version.
