"""
Generate a MOCK Non-Routine (NR) prediction artifact for the simulator.

The public project replaces the simulator's fleet-average sampled NR with a conditional,
feature-driven mock artifact. This script emits a synthetic artifact with the same
runtime schema expected by downstream simulator code.

Outputs (under Data/input/nr_prediction/):
  - nr_conditional.csv : one row per (fleet, rt_code); calibrated conditional NR-labour
                         quantiles q05..q95, plus p_nr. rt_code == '__fleet_default__'
                         is the fallback used when a slot task's code is missing.
  - nr_calibration.json: per-fleet static-mass reference + calibration metadata so the
                         predicted injection preserves the static model's *total* NR mass
                         (it changes NR *placement*, not mass -- PAPER_DESIGN sec. 4.4).

Run from the repo root:  python generate_mock_nr_artifact.py
"""

import os
import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from numpy.random import default_rng

# Keep this list in sync with the predictor (simulation/nr_predictor.py).
QUANTILE_COLUMNS = ['q05', 'q10', 'q25', 'q50', 'q75', 'q90', 'q95']
QUANTILE_LEVELS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

# Mirror anemos' NR filtering (M.MAX_DURATION_NR) without importing the heavy config.
MAX_DURATION_NR = 50.0
# Number of synthetic routine-task codes to emit per fleet (mock only; the real export
# emits one row per real JIC/RT code). The fleet-default row is added on top.
N_MOCK_RT_CODES = 12
# Reproducibility for the mock artifact.
MOCK_SEED = 20240607


def _static_mean_per_occurrence(labor_model, n=200_000, seed=MOCK_SEED):
    """Expected NR labour hours *per occurrence* under the static model, computed by
    sampling the fitted distribution exactly as anemos.__sample_from_distribution does
    (draw, drop <=0 and >MAX_DURATION_NR). This is the calibration target."""
    rng = default_rng(seed)
    sample = labor_model['distr'].rvs(*labor_model['arg'],
                                      loc=labor_model['loc'],
                                      scale=labor_model['scale'],
                                      size=n, random_state=rng)
    sample = sample[(sample > 0) & (sample <= MAX_DURATION_NR)]
    if sample.size == 0:
        # Degenerate fitted distribution -- fall back to a small positive mean.
        return 1.0
    return float(np.mean(sample))


def _lognormal_quantiles(mean, cv):
    """Quantile grid (QUANTILE_LEVELS) of a lognormal with the given arithmetic mean and
    coefficient of variation. Returns values whose *median* != mean (mean > median), which
    is what lets a risk buffer (q>0.5) add mass on purpose above the calibrated centre."""
    from scipy.stats import norm
    sigma2 = np.log(1.0 + cv ** 2)
    sigma = np.sqrt(sigma2)
    mu = np.log(max(mean, 1e-9)) - 0.5 * sigma2
    return np.exp(mu + sigma * norm.ppf(QUANTILE_LEVELS))


def build_artifact(distributions_NR, source='mock'):
    """Build the (conditional_df, calibration_dict) pair from a list of per-fleet NR
    distribution dicts (the structure anemos loads from Data/pickle/distributions_NR)."""
    rng = np.random.default_rng(MOCK_SEED)
    rows = []
    calibration = {
        'schema_version': 1,
        'source': source,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'quantile_columns': QUANTILE_COLUMNS,
        'quantile_levels': QUANTILE_LEVELS,
        'max_duration_nr': MAX_DURATION_NR,
        'notes': ('Predicted NR is gated per hangar slot with the fleet probability_NR '
                  '(identical to static), then magnitude = mean over the slot routine '
                  'tasks of their calibrated conditional quantile. Per-fleet quantiles '
                  'are calibrated so the occurrence-unweighted mean of q50 across rt_codes '
                  'equals static_mean_per_occurrence -> total NR mass is preserved; the '
                  'risk-buffer quantile (>0.5) adds mass on purpose (Exp. B).'),
        'fleets': {},
    }

    for fleet_dict in distributions_NR:
        fleet = str(fleet_dict['fleet'])
        p_nr = float(fleet_dict.get('probability_NR', 0.3))
        static_mean = _static_mean_per_occurrence(fleet_dict['labor_fitted'].model)

        # Synthesize per-rt_code conditional means with spread so NR placement varies by
        # routine task (the "conditional, feature-driven" signal). Means are drawn around
        # static_mean; we then calibrate so their q50 average lands back on static_mean.
        code_means = static_mean * rng.lognormal(mean=0.0, sigma=0.45, size=N_MOCK_RT_CODES)
        code_cvs = rng.uniform(0.5, 1.1, size=N_MOCK_RT_CODES)
        code_pnr = np.clip(p_nr * rng.uniform(0.6, 1.4, size=N_MOCK_RT_CODES), 0.01, 0.99)

        raw_q = np.array([_lognormal_quantiles(m, cv) for m, cv in zip(code_means, code_cvs)])
        # Calibrate: scale so the unweighted mean of code medians (q50) == static_mean.
        q50_idx = QUANTILE_LEVELS.index(0.50)
        mean_median = float(np.mean(raw_q[:, q50_idx]))
        scale = static_mean / mean_median if mean_median > 0 else 1.0
        cal_q = raw_q * scale

        for i in range(N_MOCK_RT_CODES):
            row = {'fleet': fleet, 'rt_code': f'{fleet}_MRI{i:03d}', 'p_nr': round(float(code_pnr[i]), 4)}
            row.update({c: round(float(cal_q[i, j]), 4) for j, c in enumerate(QUANTILE_COLUMNS)})
            rows.append(row)

        # Fleet-default fallback row: the calibrated fleet-average conditional distribution.
        default_q = _lognormal_quantiles(static_mean, cv=0.8)
        default_scale = static_mean / default_q[q50_idx] if default_q[q50_idx] > 0 else 1.0
        default_q = default_q * default_scale
        drow = {'fleet': fleet, 'rt_code': '__fleet_default__', 'p_nr': round(p_nr, 4)}
        drow.update({c: round(float(default_q[j]), 4) for j, c in enumerate(QUANTILE_COLUMNS)})
        rows.append(drow)

        # Probability-indicator calibration: scale per-code p_nr so the unweighted mean over
        # codes equals the fleet probability_NR (preserves expected NR occurrence; the prediction
        # only redistributes *which* slots are likely to generate NR).
        mean_code_p_nr = float(np.mean(code_pnr))
        prob_scale = (p_nr / mean_code_p_nr) if mean_code_p_nr > 0 else 1.0

        calibration['fleets'][fleet] = {
            'probability_NR': p_nr,
            'static_mean_per_occurrence': round(static_mean, 4),
            'static_expected_nr_per_slot': round(p_nr * static_mean, 4),
            'calibration_scale_applied': round(float(scale), 6),
            'mean_code_p_nr': round(mean_code_p_nr, 6),
            'prob_scale': round(float(prob_scale), 6),
            'n_rt_codes': N_MOCK_RT_CODES,
            'runtime_scale': 1.0,   # adjusted by the calibration guard if residual mass drifts
        }

    conditional_df = pd.DataFrame(rows)
    return conditional_df, calibration


def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    pickle_path = os.path.join(repo_root, 'Data', 'pickle', 'distributions_NR')
    out_dir = os.path.join(repo_root, 'Data', 'input', 'nr_prediction')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(pickle_path):
        raise FileNotFoundError(
            f'{pickle_path} not found. Run generate_dummy_data.py first to create the mock '
            'distributions_NR pickle (or provide the real one).')

    with open(pickle_path, 'rb') as f:
        distributions_NR = pickle.load(f)

    conditional_df, calibration = build_artifact(distributions_NR, source='mock')

    csv_path = os.path.join(out_dir, 'nr_conditional.csv')
    json_path = os.path.join(out_dir, 'nr_calibration.json')
    conditional_df.to_csv(csv_path, index=False)
    with open(json_path, 'w') as f:
        json.dump(calibration, f, indent=2)

    print(f'Wrote {csv_path}  ({len(conditional_df)} rows, '
          f'{conditional_df["fleet"].nunique()} fleets)')
    print(f'Wrote {json_path}')
    for fleet, info in calibration['fleets'].items():
        print(f"  fleet {fleet}: static_mean/occurrence={info['static_mean_per_occurrence']} "
              f"p_nr={info['probability_NR']} -> E[NR/slot]={info['static_expected_nr_per_slot']}")


if __name__ == '__main__':
    main()
