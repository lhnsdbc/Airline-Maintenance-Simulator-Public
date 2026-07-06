"""
Monitor partial or complete experiment outputs before spending the full run budget.

Reads Data/output/<scenario>_exp*/ results. If a cell has a complete_results_overview CSV,
that is used; otherwise per-iteration results_overview CSVs are loaded. The checks are
deliberately conservative: warnings are meant to trigger inspection, not prove a hypothesis.

Usage:
    python -m experiments.monitor_results --scenario default_run --expected-iterations 20
    python -m experiments.monitor_results --scenario default_run --strict
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from experiments.analysis import (_parse_tag, aggregate, calibration_summary, METRICS)
from experiments.run_experiments import EXPERIMENTS, MAIN_EXPERIMENT_ORDER


KEY_OPERATIONAL_METRICS = [
    'slots_included_in_aog',
    'completion_factor',
    'flights_delay_dep',
    'rotations_delay_dep',
    'rotations_cancelled',
    'recovery_module_disr_call_count',
]


def _output_dir(output_dir=None):
    if output_dir is not None:
        return output_dir
    from config import directories
    return directories.output


def expected_folders(scenario, exp='main'):
    if exp == 'main':
        exps = MAIN_EXPERIMENT_ORDER
    else:
        exps = [e.strip().upper() for e in exp.split(',') if e.strip()]
    folders = []
    for e in exps:
        for cell in EXPERIMENTS[e]():
            folders.append(f'{scenario}_{cell["tag"]}')
    return folders


def _read_csvs(paths):
    frames = []
    for path in paths:
        try:
            frames.append(pd.read_csv(path))
        except Exception as exc:
            print(f'WARN: could not read {path}: {exc}')
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_partial_overviews(scenario='default_run', output_dir=None):
    """Load complete cell overviews, or per-iteration overview files for in-progress cells."""
    output_dir = _output_dir(output_dir)
    frames = []
    for folder_path in sorted(glob.glob(os.path.join(output_dir, f'{scenario}_exp*'))):
        if not os.path.isdir(folder_path):
            continue
        folder = os.path.basename(folder_path)
        complete = glob.glob(os.path.join(folder_path, 'complete_results_overview_*.csv'))
        paths = complete or glob.glob(os.path.join(folder_path, 'results_overview_*.csv'))
        df = _read_csvs(paths)
        if df.empty:
            continue
        info = _parse_tag(folder)
        for k, v in info.items():
            df[k] = v
        df['run_folder'] = folder
        df['source_complete'] = bool(complete)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _iteration_count(group):
    if 'sim_iteration' in group.columns:
        return int(group['sim_iteration'].nunique())
    return int(len(group))


def cell_status(df, expected):
    rows = []
    present = set(df['run_folder'].dropna().unique()) if not df.empty else set()
    for folder in expected:
        sub = df[df['run_folder'] == folder] if not df.empty else pd.DataFrame()
        rows.append({
            'run_folder': folder,
            'present': folder in present,
            'n_iter': _iteration_count(sub) if not sub.empty else 0,
            'complete_file': bool(sub['source_complete'].any()) if 'source_complete' in sub.columns else False,
        })
    return pd.DataFrame(rows)


def _metric_available(df, metric):
    return metric in df.columns and df[metric].notna().any()


def _check_expA_proxy(agg, alerts):
    if agg.empty or 'interval_spillage_mean' not in agg.columns:
        return
    sub = agg[agg['exp'] == 'expA'] if 'exp' in agg.columns else agg
    for nr in ['static', 'predicted']:
        s = sub[sub['nr_mode'] == nr].groupby('rung')['interval_spillage_mean'].mean()
        if {'R0', 'R2'}.issubset(s.index) and not (s.loc['R2'] < s.loc['R0']):
            alerts.append(f'Exp A {nr}: interval_spillage did not improve R0->R2 '
                          f'({s.loc["R0"]:.4g} -> {s.loc["R2"]:.4g}).')


def _check_flat_policy_signal(agg, alerts):
    if agg.empty:
        return
    sub = agg[agg['exp'] == 'expA'] if 'exp' in agg.columns else agg
    for nr in ['static', 'predicted']:
        mode = sub[sub['nr_mode'] == nr]
        for metric in KEY_OPERATIONAL_METRICS:
            col = f'{metric}_mean'
            if col not in mode.columns:
                continue
            vals = mode.groupby('rung')[col].mean().dropna()
            if len(vals) >= 2 and np.isclose(vals.max(), vals.min()):
                alerts.append(f'Exp A {nr}: {metric} is identical across available rungs.')


def _check_static_predicted_identical(agg, alerts):
    if agg.empty:
        return
    sub = agg[agg['exp'] == 'expA'] if 'exp' in agg.columns else agg
    for metric in KEY_OPERATIONAL_METRICS:
        col = f'{metric}_mean'
        if col not in sub.columns:
            continue
        piv = sub.pivot_table(index='rung', columns='nr_mode', values=col, aggfunc='mean')
        if {'static', 'predicted'}.issubset(piv.columns):
            diff = (piv['predicted'] - piv['static']).dropna()
            if len(diff) >= 2 and np.allclose(diff.values, 0):
                alerts.append(f'Exp A: predicted and static are identical for {metric}.')


def _check_calibration(scenario, output_dir, alerts):
    cal = calibration_summary(scenario, output_dir)
    if cal.empty:
        alerts.append('No NR calibration files found yet; expected after predicted cells complete iterations.')
        return cal
    if 'prob_ratio_pred_over_static' in cal.columns:
        bad = cal['prob_ratio_pred_over_static'].dropna()
        bad = bad[(bad < 0.95) | (bad > 1.05)]
        if not bad.empty:
            alerts.append(f'NR probability calibration drift outside [0.95, 1.05] in {len(bad)} row(s).')
    if {'n_injections', 'mass_ratio_pred_over_static'}.issubset(cal.columns):
        mass = cal[pd.to_numeric(cal['n_injections'], errors='coerce').fillna(0) > 0]
        bad = mass['mass_ratio_pred_over_static'].dropna()
        bad = bad[(bad < 0.5) | (bad > 1.5)]
        if not bad.empty:
            alerts.append(f'NR mass ratio outside [0.5, 1.5] after positive injections in {len(bad)} row(s).')
    return cal


def monitor(scenario='default_run', output_dir=None, expected_iterations=20, exp='main',
            require_complete=False):
    output_dir = _output_dir(output_dir)
    expected = expected_folders(scenario, exp)
    df = load_partial_overviews(scenario, output_dir)
    alerts = []

    print(f'=== Monitor {scenario} ({exp}) ===')
    print(f'Output dir: {output_dir}')
    print(f'Expected cells: {len(expected)}')

    status = cell_status(df, expected)
    if status.empty:
        print('No matching experiment outputs found.')
        return 1

    print('\n=== Cell status ===')
    print(status.to_string(index=False))

    missing = status[~status['present']]
    short = status[(status['present']) & (status['n_iter'] < expected_iterations)]
    if not missing.empty:
        print(f'\nPending/missing cells: {len(missing)}')
    if not short.empty:
        print(f'Cells below {expected_iterations} iteration(s): {len(short)}')
    if require_complete and (not missing.empty or not short.empty):
        alerts.append('Required complete grid is not complete.')

    if df.empty:
        print('\nNo overview rows loaded yet.')
        return 1 if alerts else 0

    metrics = [m for m in METRICS if _metric_available(df, m)]
    agg = aggregate(df, metrics=metrics)
    print('\n=== KPI snapshot ===')
    cols = ['exp', 'rung', 'nr_mode', 'buffer_q', 'var_scale', 'n_iter']
    metric_cols = [f'{m}_mean' for m in metrics if f'{m}_mean' in agg.columns]
    print(agg[[c for c in cols + metric_cols if c in agg.columns]].to_string(index=False))

    _check_expA_proxy(agg, alerts)
    _check_flat_policy_signal(agg, alerts)
    _check_static_predicted_identical(agg, alerts)
    cal = _check_calibration(scenario, output_dir, alerts)
    if not cal.empty:
        print('\n=== NR calibration snapshot ===')
        print(cal.to_string(index=False))

    print('\n=== Monitoring verdict ===')
    if alerts:
        print('Inspect before continuing:')
        for alert in alerts:
            print(f'  - {alert}')
        return 1
    print('No blocking monitoring alerts from available outputs.')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description='Monitor partial experiment outputs.')
    ap.add_argument('--scenario', default='default_run')
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--expected-iterations', type=int, default=20)
    ap.add_argument('--exp', default='main',
                    help="Expected experiment set: 'main' or comma-separated letters such as A,B,D.")
    ap.add_argument('--require-complete', action='store_true',
                    help='Treat missing/short cells as blocking alerts.')
    args = ap.parse_args(argv)
    return monitor(args.scenario, args.output_dir, args.expected_iterations,
                   args.exp, args.require_complete)


if __name__ == '__main__':
    raise SystemExit(main())
