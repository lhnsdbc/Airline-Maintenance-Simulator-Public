# Case Study: Reserving Non-Routine Maintenance Capacity

> This case study uses only public synthetic data and synthetic KPI proxies.

## Operational question

How should a maintenance planning team compare a conservative schedule with a policy that anticipates non-routine (NR) maintenance, while making the service and maintenance-capacity trade-off visible?

## Simulation and policy approach

The simulator packages a seven-day synthetic fleet profile, produces reproducible KPI artifacts for policy rungs, and exposes them through a Dash comparison screen and FastAPI service. The comparison holds the NR forecast mode fixed at `predicted`, then contrasts the R0 strict-baseline proxy with the R2 learning-inspired proxy. The grounded analyst report reads the same KPI artifact shown in the dashboard, so its narrative is traceable to run IDs and values.

## Synthetic evaluation design

The evaluation uses three deterministic synthetic stress profiles—nominal, high utilisation, and constrained hangar—over five fixed seeds. That produces 30 observations, reported as median and P10--P90 distributions rather than a single best run. The full protocol, KPI definitions, and results are in the [evaluation card](EVALUATION.md).

## Key trade-off

The synthetic R2 proxy shows lower delay and interval-spillage proxies in each profile, while executing fewer A-check slots. In the nominal profile, the medians are 3513.65 versus 4242.87 for delay proxy and 2.52 versus 3.00 for A-check slots. The decision is therefore not framed as “best score wins”: it asks whether the apparent service gain is acceptable given lower planned maintenance capacity.

## Limitation

These effects are engineered by public synthetic factors; R2 is not a disclosed trained model and the score is not an airline KPI. The artifact is useful for discussing evaluation design, traceable reporting, and stakeholder trade-offs—not for operational recommendation or a claim of predictive accuracy.
