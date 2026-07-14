# Synthetic Policy Evaluation Card

> **Scope:** all results on this page are **synthetic policy-proxy results**. They demonstrate a repeatable evaluation workflow and are not evidence of performance for a trained production policy or a real airline.

## Decision question

Under nominal, high-utilisation, and constrained-hangar synthetic conditions, how does the strict baseline policy proxy (R0) compare with the learning-inspired policy proxy (R2) when both use the same predicted non-routine (NR) workload signal?

## Evaluation design

The generated `default_run` fixture contains 31 aircraft, 38 airports, 127 rotations, and 311 flight legs over seven synthetic days. The protocol evaluates two policy rungs across five fixed seeds (`20260706`--`20260710`) and holds NR mode at `predicted` for both policies. This yields 30 observations: 3 scenarios × 5 seeds × 2 policies.

| Policy proxy | Implementation rung | Interpretation |
| --- | --- | --- |
| Strict baseline | R0 / `strict_baseline` | Reference scheduling heuristic proxy. |
| Learning-inspired policy proxy | R2 / `free_optimizer` | Synthetic improvement proxy; it is **not** a reported trained-policy result. |

The named stress scenarios are deterministic multipliers over the same public fixture:

| Scenario | Added synthetic condition |
| --- | --- |
| Nominal | No additional multiplier. |
| High utilisation | Delay ×1.18, spillage ×1.08, A-check slots ×0.95. |
| Constrained hangar | Delay ×1.08, spillage ×1.28, A-check slots ×0.84. |

## KPI definitions

| KPI | Direction | Definition in this public evaluation |
| --- | --- | --- |
| Policy quality score | Higher | `100 - 0.015 × departure-delay proxy - 0.85 × interval-spillage proxy - 1.7 × uncovered-NR-labour-hours`. A synthetic composite, not a monetary or operational KPI. |
| Departure-delay proxy | Lower | Aggregated synthetic delay proxy across the fixture’s flight legs; units are deliberately not presented as airline minutes. |
| Interval-spillage proxy | Lower | Synthetic proxy for maintenance-interval pressure. |
| A-check slots executed | Context / trade-off | Synthetic count of usable A-check slot templates executed. |
| Uncovered NR labour | Lower | Synthetic realised NR workload minus reserved NR workload; held within the composite but not used to claim real staffing requirements. |

## Distribution results

Each cell reports **median [P10, P90]** across five fixed seeds. Values below were generated with the reproducibility command in this card; precision is shown only to make the calculation auditable.

| Scenario | Policy proxy | Quality score | Delay proxy | Spillage proxy | A-check slots |
| --- | --- | ---: | ---: | ---: | ---: |
| Nominal | Strict baseline | 13.23 [10.27, 13.56] | 4242.87 [4226.73, 4387.66] | 27.21 [27.10, 28.14] | 3.00 [3.00, 3.00] |
| Nominal | Learning-inspired proxy | 32.21 [31.80, 34.27] | 3513.65 [3406.84, 3534.82] | 17.74 [17.21, 17.85] | 2.52 [2.52, 2.52] |
| High utilisation | Strict baseline | -0.08 [-3.49, 0.31] | 5006.59 [4987.54, 5177.44] | 29.39 [29.27, 30.39] | 2.85 [2.85, 2.85] |
| High utilisation | Learning-inspired proxy | 21.52 [21.05, 23.91] | 4146.11 [4020.07, 4171.09] | 19.16 [18.58, 19.28] | 2.39 [2.39, 2.39] |
| Constrained hangar | Strict baseline | 1.66 [-1.69, 2.04] | 4582.30 [4564.87, 4738.67] | 34.83 [34.69, 36.01] | 2.52 [2.52, 2.52] |
| Constrained hangar | Learning-inspired proxy | 23.77 [23.31, 26.09] | 3794.74 [3679.39, 3817.61] | 22.71 [22.02, 22.85] | 2.12 [2.12, 2.12] |

## Readout and trade-off

In this deliberately constructed proxy, R2 has lower delay and spillage across every tested scenario. The visible trade-off is lower A-check slot execution: in the nominal profile it uses 2.52 slots versus the baseline’s 3.00. That behaviour is built into the synthetic rung factors, so it should be read as a demonstration of how the dashboard and report surface trade-offs—not as evidence that a learned method dominates a baseline in the field.

## Reproduce

Generate the public fixture first, then run the fixed protocol:

```powershell
py generate_dummy_data.py
py generate_mock_nr_artifact.py
py -m experiments.evaluation --seeds 20260706 20260707 20260708 20260709 20260710
```

The command writes `runs.csv`, `summary.csv`, and `protocol.json` to `artifacts/evaluation_card/`. `protocol.json` records the fixed seeds, scenario multipliers, policy labels, and the constant NR mode.

## Limitations

- The fixture, stress multipliers, KPI factors, and policy effects are synthetic and deterministic; they are not calibrated against observed airline data.
- Five seeds demonstrate repeatability and spread, not statistical power or a claim of significance.
- R2 is a learning-inspired proxy backed by the public experiment scaffold, not a released trained model or offline-RL evaluation.
- The composite score encodes chosen weights and omits safety, crew, passenger, network, and financial consequences.
- A future real evaluation would need locked historical splits, operationally meaningful units, model-versioned predictions, uncertainty analysis, and review by domain owners.
