"""
PAPER_DESIGN.md analysis (Phase 5).

Aggregates the per-cell simulation outputs written by experiments/run_experiments.py
(Data/output/<scenario>_<tag>/complete_results_overview_*.csv), computes KPI means with
confidence intervals, paired non-parametric tests (Wilcoxon) across NR modes per rung,
policy contrasts, and the NR mass-preservation calibration summary.

Pure pandas/numpy/scipy -- no simulator dependency. Run after a batch:
    python -m experiments.analysis --scenario default_run
"""

import os
import re
import glob
import argparse

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception:  # scipy optional for pure-aggregation use
    wilcoxon = None

# KPI columns from results_overview, with the direction that means "better".
# (Cost is intentionally excluded: the only cost model in the codebase is a partial
# passenger-disruption cost that is not wired into this pipeline. See EXPERIMENT_PLAN.md.)
METRICS = {
    # Availability
    'slots_included_in_aog': {'label': 'AOG (slots in AOG)', 'better': 'lower'},
    'completion_factor':     {'label': 'Completion factor', 'better': 'higher'},
    # Punctuality
    'flights_delay_dep':     {'label': 'Departure delay (flights)', 'better': 'lower'},
    'rotations_delay_dep':   {'label': 'Departure delay (rotations)', 'better': 'lower'},
    'flights_delay_arr':     {'label': 'Arrival delay (flights)', 'better': 'lower'},
    'rotations_cancelled':   {'label': 'Rotations cancelled', 'better': 'lower'},
    # Maintenance
    'interval_spillage':     {'label': 'Interval spillage (%)', 'better': 'lower'},   # representative internal proxy; Paper-2 objective is composite (spillage, panel reuse, workload variance)
    'tasks_missed':          {'label': 'Drop-out tasks', 'better': 'lower'},
    'drop_out_task_share':   {'label': 'Drop-out task share (%)', 'better': 'lower'},
    'tasks_execution_factor': {'label': 'Tasks executed (share)', 'better': 'higher'},
    'a_check_occupation_hours': {'label': 'Executed A-check occupation (hours)', 'better': 'lower'},
    'a_check_slots_executed': {'label': 'Executed A-check visits', 'better': 'lower'},
    'nr_reserved_hours':      {'label': 'Reserved NR labour (hours)', 'better': 'lower'},
    'nr_realized_hours':      {'label': 'Realized NR labour (hours)', 'better': 'lower'},
    'nr_uncovered_labor_hours': {'label': 'Uncovered NR labour (hours)', 'better': 'lower'},
    'nr_reserve_realized_corr': {'label': 'Reserve-realized NR correlation', 'better': 'higher'},
    'nr_overrun_hours':      {'label': 'NR overrun (labour hours)', 'better': 'lower'},
    # Recovery effort
    'aircraft_swaps':        {'label': 'Aircraft swaps', 'better': 'lower'},
    'recovery_module_disr_call_count': {'label': 'Recovery calls (disruption)', 'better': 'lower'},
}

_TAG_RE = re.compile(r'_(exp[A-F])_(.+)$')


def _parse_tag(folder_name):
    """Extract (exp, cell_name, rung, nr_mode, buffer_q, var_scale, reserves) from a folder."""
    m = _TAG_RE.search(folder_name)
    info = {'exp': None, 'cell': folder_name, 'rung': None, 'nr_mode': None,
            'buffer_q': None, 'var_scale': None, 'reserves': None}
    if m:
        info['exp'] = m.group(1)
        cell = m.group(2)
        info['cell'] = cell
        rm = re.search(r'(R[012])', cell)
        if rm:
            info['rung'] = rm.group(1)
        if 'predicted' in cell or '_pred' in cell:
            info['nr_mode'] = 'predicted'
        elif 'static' in cell:
            info['nr_mode'] = 'static'
        for key, pat in (('buffer_q', r'q([0-9.]+)'), ('var_scale', r'v([0-9.]+)'),
                         ('reserves', r'res([0-9]+)')):
            pm = re.search(pat, cell)
            if pm:
                info[key] = float(pm.group(1))
    return info


def load_overviews(scenario='default_run', output_dir=None, run_suffix=None):
    """Load and tag every per-cell overview CSV into one long DataFrame (one row/iteration)."""
    if output_dir is None:
        from config import directories
        output_dir = directories.output
    frames = []
    pattern = os.path.join(output_dir, f'{scenario}_*', 'complete_results_overview_*.csv')
    for path in glob.glob(pattern):
        folder = os.path.basename(os.path.dirname(path))
        if run_suffix and not folder.endswith(run_suffix):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        # The overview's legacy ground_time_hours includes week-long LM availability
        # windows. Derive defensible optimiser-structure KPIs from genuinely used,
        # executed A-check slots instead.
        slot_paths = glob.glob(os.path.join(
            os.path.dirname(path), 'complete_results_slots_*.csv'))
        if slot_paths:
            try:
                slots = pd.read_csv(slot_paths[0], low_memory=False)
                used_a = slots[
                    (slots['slot_type'] == 'A')
                    & (slots['execution_state'] == 'executed')
                    & (pd.to_numeric(slots['work_sched_labor'], errors='coerce').fillna(0) > 0)
                ].copy()
                if not used_a.empty:
                    used_a['duration_act'] = pd.to_numeric(
                        used_a['duration_act'], errors='coerce').fillna(0)
                    a_metrics = used_a.groupby('iteration').agg(
                        a_check_occupation_hours=('duration_act', lambda s: s.sum() / 60.0),
                        a_check_slots_executed=('id', 'size'),
                    )
                    df = df.merge(
                        a_metrics, left_on='sim_iteration', right_index=True, how='left')
            except Exception:
                pass
        info = _parse_tag(folder)
        for k, v in info.items():
            df[k] = v
        df['run_folder'] = folder
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _ci95(series):
    s = series.dropna().astype(float)
    n = len(s)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    mean = s.mean()
    if n < 2:
        return (mean, np.nan, np.nan)
    sem = s.std(ddof=1) / np.sqrt(n)
    return (mean, mean - 1.96 * sem, mean + 1.96 * sem)


def aggregate(df, metrics=None, group_cols=('exp', 'rung', 'nr_mode', 'buffer_q', 'var_scale', 'reserves')):
    """Per-cell mean + 95% CI for each metric present in the data."""
    if df.empty:
        return pd.DataFrame()
    metrics = metrics or [m for m in METRICS if m in df.columns]
    group_cols = [c for c in group_cols if c in df.columns]
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row['n_iter'] = len(g)
        for metric in metrics:
            mean, lo, hi = _ci95(g[metric])
            row[f'{metric}_mean'] = mean
            row[f'{metric}_ci_lo'] = lo
            row[f'{metric}_ci_hi'] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def wilcoxon_static_vs_predicted(df, metric, rung=None, exp='expA'):
    """Paired Wilcoxon (static vs predicted) per rung, pairing on sim_iteration.
    Scoped to a single experiment (default expA, the ladder) so each rung has exactly one
    static/predicted cell pair. Returns (rung, n_pairs, statistic, p_value, median_diff)."""
    if df.empty or wilcoxon is None or 'nr_mode' not in df.columns:
        return pd.DataFrame()
    if exp is not None and 'exp' in df.columns:
        df = df[df['exp'] == exp]
    if df.empty:
        return pd.DataFrame()
    results = []
    rungs = [rung] if rung else sorted(df['rung'].dropna().unique())
    pair_key = 'sim_iteration' if 'sim_iteration' in df.columns else None
    for r in rungs:
        sub = df[df['rung'] == r]
        a = sub[sub['nr_mode'] == 'static']
        b = sub[sub['nr_mode'] == 'predicted']
        if pair_key:
            merged = a.merge(b, on=pair_key, suffixes=('_s', '_p'))
            xs, ys = merged[f'{metric}_s'], merged[f'{metric}_p']
        else:
            n = min(len(a), len(b))
            xs, ys = a[metric].values[:n], b[metric].values[:n]
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        mask = ~(np.isnan(xs) | np.isnan(ys))
        xs, ys = xs[mask], ys[mask]
        if len(xs) < 1 or np.allclose(xs, ys):
            results.append({'rung': r, 'metric': metric, 'n_pairs': len(xs),
                            'statistic': np.nan, 'p_value': np.nan,
                            'median_diff_pred_minus_static': float(np.median(ys - xs)) if len(xs) else np.nan})
            continue
        try:
            stat, p = wilcoxon(xs, ys)
        except ValueError:
            stat, p = np.nan, np.nan
        results.append({'rung': r, 'metric': metric, 'n_pairs': len(xs),
                        'statistic': stat, 'p_value': p,
                        'median_diff_pred_minus_static': float(np.median(ys - xs))})
    return pd.DataFrame(results)


def find_breakpoint(agg, metric, nr_mode):
    """On the rung ladder (R0->R1->R2), the breakpoint is the first rung at which the KPI
    stops improving (turns). Returns the rung label or None (monotone improvement)."""
    order = ['R0', 'R1', 'R2']
    sub = agg[agg['nr_mode'] == nr_mode]
    if 'exp' in sub.columns:
        sub = sub[sub['exp'] == 'expA']   # the deregulation-ladder experiment only
    col = f'{metric}_mean'
    if col not in sub.columns:
        return None
    # Mean per rung (defensive against >1 row per rung).
    per_rung = sub.groupby('rung')[col].mean()
    vals = [per_rung.get(r, np.nan) for r in order]
    better = METRICS.get(metric, {}).get('better', 'lower')
    improve = (lambda prev, cur: cur < prev) if better == 'lower' else (lambda prev, cur: cur > prev)
    for i in range(1, len(vals)):
        if np.isnan(vals[i]) or np.isnan(vals[i - 1]):
            continue
        if not improve(vals[i - 1], vals[i]):
            return order[i - 1]   # improvement stopped after the previous rung
    return None


def calibration_summary(scenario='default_run', output_dir=None, run_suffix=None):
    """Aggregate the NR mass-preservation reports (results_nr_calibration_*.csv)."""
    if output_dir is None:
        from config import directories
        output_dir = directories.output
    frames = []
    pattern = os.path.join(output_dir, f'{scenario}_*', 'results_nr_calibration_*.csv')
    for path in glob.glob(pattern):
        if run_suffix and not os.path.basename(os.path.dirname(path)).endswith(run_suffix):
            continue
        try:
            df = pd.read_csv(path)
            df['run_folder'] = os.path.basename(os.path.dirname(path))
            frames.append(df)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_report(scenario='default_run', output_dir=None, run_suffix=None):
    df = load_overviews(scenario, output_dir, run_suffix=run_suffix)
    if df.empty:
        print(f'No overview outputs found for scenario "{scenario}". Run experiments first.')
        return df, pd.DataFrame()
    agg = aggregate(df)
    print('=== Per-cell KPI aggregates ===')
    print(agg.to_string(index=False))
    print('\n=== Paired Wilcoxon (static vs predicted) ===')
    for metric in [m for m in METRICS if m in df.columns]:
        w = wilcoxon_static_vs_predicted(df, metric)
        if not w.empty:
            print(w.to_string(index=False))
    print('\n=== Experiment A policy means (R0 -> R1 -> R2) ===')
    if 'exp' in agg.columns:
        cols = ['rung', 'nr_mode'] + [
            f'{m}_mean' for m in METRICS if f'{m}_mean' in agg.columns]
        print(agg[agg['exp'] == 'expA'][cols].sort_values(
            ['nr_mode', 'rung']).to_string(index=False))
    cal = calibration_summary(scenario, output_dir, run_suffix=run_suffix)
    if not cal.empty:
        print('\n=== NR mass-preservation (sec. 4.4) ===')
        print(cal.to_string(index=False))
    return df, agg


def _parse_rung_suffixes(default_suffix, per_rung_suffixes=None):
    suffixes = {rung: default_suffix for rung in ['R0', 'R1', 'R2']}
    if per_rung_suffixes:
        for item in per_rung_suffixes.split(','):
            if not item.strip():
                continue
            rung, suffix = item.split(':', 1)
            suffixes[rung.strip()] = suffix.strip()
    return suffixes


def budget_matched_report(scenario='default_run', output_dir=None,
                          reference_suffix='_updated5_20260625',
                          reference_suffixes=None,
                          budget_suffix='',
                          iterations=(0, 1, 2, 3, 4)):
    """Compare Experiment F budget-matched static cells against Experiment A predicted cells."""
    df = load_overviews(scenario, output_dir)
    if df.empty:
        print(f'No overview outputs found for scenario "{scenario}". Run experiments first.')
        return pd.DataFrame()

    metrics = ['nr_reserved_hours', 'nr_uncovered_labor_hours', 'nr_reserve_realized_corr']
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise SystemExit('Missing required KPI columns: ' + ', '.join(missing))

    rows = []
    reference_by_rung = _parse_rung_suffixes(reference_suffix, reference_suffixes)
    for rung in ['R0', 'R1', 'R2']:
        pred_folder = f'{scenario}_expA_{rung}_predicted{reference_by_rung[rung]}'
        matched_folder = f'{scenario}_expF_{rung}_static_budgetmatched{budget_suffix}'
        pred = df[df['run_folder'] == pred_folder]
        matched = df[df['run_folder'] == matched_folder]
        if iterations is not None:
            pred = pred[pred['sim_iteration'].isin(iterations)]
            matched = matched[matched['sim_iteration'].isin(iterations)]
        if pred.empty or matched.empty:
            missing = []
            if pred.empty:
                missing.append(pred_folder)
            if matched.empty:
                missing.append(matched_folder)
            rows.append({'rung': rung, 'status': 'missing: ' + ', '.join(missing)})
            continue
        row = {'rung': rung, 'status': 'ok', 'n_predicted': len(pred), 'n_budget_static': len(matched)}
        for metric in metrics:
            p_mean = pred[metric].astype(float).mean()
            s_mean = matched[metric].astype(float).mean()
            row[f'predicted_{metric}'] = p_mean
            row[f'budget_static_{metric}'] = s_mean
            row[f'delta_static_minus_predicted_{metric}'] = s_mean - p_mean
        row['budget_within_5pct'] = (
            abs(row['delta_static_minus_predicted_nr_reserved_hours'])
            <= 0.05 * abs(row['predicted_nr_reserved_hours'])
        )
        row['supports_targeting_claim'] = (
            row['budget_within_5pct']
            and row['budget_static_nr_uncovered_labor_hours'] > row['predicted_nr_uncovered_labor_hours']
            and row['budget_static_nr_reserve_realized_corr'] < row['predicted_nr_reserve_realized_corr']
        )
        rows.append(row)

    report = pd.DataFrame(rows)
    print('=== Experiment F budget-matched NR report ===')
    print(report.to_string(index=False))
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='default_run')
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--run-suffix', default=None,
                    help='Only load output folders ending with this suffix.')
    ap.add_argument('--budget-report', action='store_true',
                    help='Compare expF budget-matched static cells against expA predicted cells.')
    ap.add_argument('--budget-reference-suffix', default='_updated5_20260625',
                    help='ExpA predicted folder suffix used as the budget-report reference.')
    ap.add_argument('--budget-reference-suffixes', default=None,
                    help='Optional per-rung ExpA suffixes, e.g. '
                         'R0:_gate75_20260627,R1:_updated5_20260625,R2:_updated5_20260625.')
    ap.add_argument('--budget-suffix', default='',
                    help='Suffix appended to expF budget-matched folders.')
    args = ap.parse_args()
    if args.budget_report:
        budget_matched_report(args.scenario, args.output_dir, args.budget_reference_suffix,
                              args.budget_reference_suffixes, args.budget_suffix)
    else:
        run_report(args.scenario, args.output_dir, args.run_suffix)
