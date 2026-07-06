"""
PAPER_DESIGN.md figures (Phase 5).

Renders the paper's data-driven figures from the aggregated experiment outputs:
  Fig 3 optimiser deployment     : maintenance and airline KPIs across R0/R1/R2.
  Fig 4 buffer frontier          : reserved capacity vs uncovered NR across buffer quantiles.
  Fig 5 uncertainty sensitivity  : policy performance vs NR variance scale.
Figs 1-2 (framework / ladder schematics) are drawn as simple box diagrams.

matplotlib only. Run after a batch + analysis:
    python -m experiments.figures --scenario default_run
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.analysis import (load_overviews, aggregate, METRICS, find_breakpoint)

RUNG_ORDER = ['R0', 'R1', 'R2']
NR_COLORS = {'static': '#888888', 'predicted': '#1f77b4'}


def _with_derived_metrics(df):
    df = df.copy()
    if {'tasks_missed', 'tasks_total'}.issubset(df.columns):
        denom = df['tasks_total'].replace(0, np.nan).astype(float)
        df['drop_out_task_share'] = df['tasks_missed'].astype(float) / denom * 100.0
    return df


def _outdir(output_dir=None):
    if output_dir is None:
        from config import directories
        output_dir = directories.output
    d = os.path.join(output_dir, 'figures')
    os.makedirs(d, exist_ok=True)
    return d


def fig_breakpoint(df, agg, metric='slots_included_in_aog', outdir='.'):
    """Fig 3: KPI vs deregulation rung, one curve per NR mode, breakpoint annotated."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(RUNG_ORDER))
    agg_a = agg[agg['exp'] == 'expA'] if 'exp' in agg.columns else agg
    for nr in ['static', 'predicted']:
        sub = agg_a[agg_a['nr_mode'] == nr].groupby('rung').mean(numeric_only=True)
        means = [sub.loc[r, f'{metric}_mean'] if r in sub.index else np.nan for r in RUNG_ORDER]
        los = [sub.loc[r, f'{metric}_ci_lo'] if r in sub.index else np.nan for r in RUNG_ORDER]
        his = [sub.loc[r, f'{metric}_ci_hi'] if r in sub.index else np.nan for r in RUNG_ORDER]
        means = np.array(means, float)
        yerr = np.array([means - np.array(los, float), np.array(his, float) - means])
        ax.errorbar(x, means, yerr=yerr, marker='o', capsize=3, color=NR_COLORS[nr], label=nr)
        bp = find_breakpoint(agg, metric, nr)
        if bp in RUNG_ORDER:
            bx = RUNG_ORDER.index(bp)
            ax.axvline(bx, color=NR_COLORS[nr], ls='--', alpha=0.4)
            ax.annotate(f'{nr} breakpoint', (bx, ax.get_ylim()[1]),
                        color=NR_COLORS[nr], fontsize=8, ha='center', va='bottom')
    ax.set_xticks(x); ax.set_xticklabels(RUNG_ORDER)
    ax.set_xlabel('Deregulation rung (strict -> free)')
    ax.set_ylabel(METRICS.get(metric, {}).get('label', metric))
    ax.set_title('Fig 3 - Breakpoint: KPI vs deregulation rung')
    ax.legend(title='NR mode')
    fig.tight_layout()
    path = os.path.join(outdir, f'fig3_breakpoint_{metric}.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def fig_scissors(df, agg, proxy='interval_spillage', ops='slots_included_in_aog',
                 nr_mode='static', outdir='.'):
    """Fig 3 (headline scissors): the optimiser's proxy (interval spillage, expected to fall) against
    an operational KPI (AOG, expected to turn) across the policy steps, on twin y-axes. The diverging
    'scissors' between the two curves is the paper's contribution."""
    sub = agg[agg['exp'] == 'expA'] if 'exp' in agg.columns else agg
    sub = sub[sub['nr_mode'] == nr_mode]
    if sub.empty or f'{proxy}_mean' not in sub.columns or f'{ops}_mean' not in sub.columns:
        return None
    g = sub.groupby('rung').mean(numeric_only=True)
    x = list(range(len(RUNG_ORDER)))
    pvals = [g.loc[r, f'{proxy}_mean'] if r in g.index else np.nan for r in RUNG_ORDER]
    ovals = [g.loc[r, f'{ops}_mean'] if r in g.index else np.nan for r in RUNG_ORDER]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()
    plabel = METRICS.get(proxy, {}).get('label', proxy)
    olabel = METRICS.get(ops, {}).get('label', ops)
    l1, = ax1.plot(x, pvals, marker='o', color='#1f77b4', label=f'{plabel} (proxy)')
    l2, = ax2.plot(x, ovals, marker='s', color='#d62728', label=f'{olabel} (operations)')
    ax1.set_xticks(x); ax1.set_xticklabels(['strict', 'moderate', 'free'])
    ax1.set_xlabel('Maintenance policy (strict -> free)')
    ax1.set_ylabel(plabel, color='#1f77b4')
    ax2.set_ylabel(olabel, color='#d62728')
    ax1.set_title(f'Fig 3 - Proxy vs operations scissors ({nr_mode} NR)')
    ax1.legend(handles=[l1, l2], loc='best', fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, 'fig3_scissors.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def fig_optimizer_panel(df, agg, outdir='.'):
    """Fig 3: optimiser-centred KPI panel across R0/R1/R2 for both NR modes."""
    metrics = [
        'interval_spillage', 'drop_out_task_share', 'nr_uncovered_labor_hours',
        'flights_delay_dep',
    ]
    available = [m for m in metrics if f'{m}_mean' in agg.columns]
    if not available:
        return None
    sub = agg[agg['exp'] == 'expA'] if 'exp' in agg.columns else agg
    # Choose a near-square grid: 2 columns when <=4 panels (2x2), 3 otherwise.
    ncols = 2 if len(available) <= 4 else 3
    nrows = -(-len(available) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.2 * nrows),
                             squeeze=False)
    x = np.arange(len(RUNG_ORDER))
    for ax, metric in zip(axes.flat, available):
        for nr in ['static', 'predicted']:
            g = sub[sub['nr_mode'] == nr].groupby('rung').mean(numeric_only=True)
            vals = [g.loc[r, f'{metric}_mean'] if r in g.index else np.nan
                    for r in RUNG_ORDER]
            ax.plot(x, vals, marker='o', color=NR_COLORS[nr], label=nr)
        ax.set_xticks(x)
        ax.set_xticklabels(RUNG_ORDER)
        ax.set_title(METRICS.get(metric, {}).get('label', metric), fontsize=9)
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(available):]:
        ax.axis('off')
    axes.flat[0].legend(title='NR mode', fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, 'fig3_optimizer_panel.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_buffer_pareto(df, agg, avail_metric='nr_reserved_hours',
                      disr_metric='nr_uncovered_labor_hours', outdir='.'):
    """Fig 4: capacity-coverage frontier across buffer quantiles (Exp. B)."""
    sub = agg[agg['exp'] == 'expB'] if 'exp' in agg.columns else agg
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    pred = sub[sub['nr_mode'] == 'predicted']
    ax.scatter(pred[f'{avail_metric}_mean'], pred[f'{disr_metric}_mean'],
               c=pred['buffer_q'], cmap='viridis', s=50, label='predicted (by buffer q)')
    for _, row in pred.iterrows():
        ax.annotate(f"q{row['buffer_q']}", (row[f'{avail_metric}_mean'], row[f'{disr_metric}_mean']),
                    fontsize=7)
    stat = sub[sub['nr_mode'] == 'static']
    if not stat.empty:
        ax.scatter(stat[f'{avail_metric}_mean'], stat[f'{disr_metric}_mean'],
                   color='red', marker='x', s=70, label='static')
    ax.set_xlabel(METRICS.get(avail_metric, {}).get('label', avail_metric))
    ax.set_ylabel(METRICS.get(disr_metric, {}).get('label', disr_metric))
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, 'fig4_buffer_pareto.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def fig_robustness(df, agg, metric='nr_uncovered_labor_hours', outdir='.'):
    """Fig 5: policy and NR-mode sensitivity to workload dispersion (Exp. D)."""
    sub = agg[agg['exp'] == 'expD'] if 'exp' in agg.columns else agg
    if sub.empty or 'var_scale' not in sub.columns:
        return None
    fig, axes = plt.subplots(1, len(RUNG_ORDER), figsize=(9.2, 3.4), sharey=False)
    markers = {'static': 'o', 'predicted': 's'}
    labels = {'static': 'static', 'predicted': 'predicted'}
    y_label = METRICS.get(metric, {}).get('label', metric)
    for ax, rung in zip(axes, RUNG_ORDER):
        for nr in ['static', 'predicted']:
            s = sub[(sub['rung'] == rung) & (sub['nr_mode'] == nr)].groupby(
                'var_scale')[f'{metric}_mean'].mean().sort_index()
            if s.empty:
                continue
            ax.plot(s.index, s.values, marker=markers[nr], ms=5.5, lw=2.0,
                    color=NR_COLORS[nr], label=labels[nr])
            ax.annotate(labels[nr], (s.index[-1], s.values[-1]), xytext=(-5, 0),
                        textcoords='offset points', va='center', ha='right', fontsize=7,
                        color=NR_COLORS[nr])
        ax.set_title(rung, fontsize=10, weight='bold')
        ax.set_xlabel('NR workload variation')
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, loc='upper left', frameon=True)
    fig.tight_layout()
    path = os.path.join(outdir, f'fig5_robustness_{metric}.png')
    fig.savefig(path, dpi=300); plt.close(fig)
    return path


def fig_budget_matched(budget_csv=None, outdir='.'):
    """Fig 6: budget-matched static reserve vs predicted reserve targeting."""
    if not budget_csv or not os.path.exists(budget_csv):
        return None
    rep = pd.read_csv(budget_csv)
    rep = rep[rep.get('status', 'ok').eq('ok')] if 'status' in rep.columns else rep
    required = {
        'rung',
        'predicted_nr_uncovered_labor_hours',
        'budget_static_nr_uncovered_labor_hours',
        'predicted_nr_reserve_realized_corr',
        'budget_static_nr_reserve_realized_corr',
    }
    if rep.empty or not required.issubset(rep.columns):
        return None

    order = [r for r in RUNG_ORDER if r in set(rep['rung'])]
    rep = rep.set_index('rung').loc[order].reset_index()
    x = np.arange(len(rep))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].bar(x - width / 2, rep['budget_static_nr_uncovered_labor_hours'],
                width, color=NR_COLORS['static'], label='budget-matched static')
    axes[0].bar(x + width / 2, rep['predicted_nr_uncovered_labor_hours'],
                width, color=NR_COLORS['predicted'], label='predicted')
    axes[0].set_title('Uncovered NR labour')
    axes[0].set_ylabel('Hours')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(rep['rung'])
    axes[0].grid(axis='y', alpha=0.2)

    axes[1].bar(x - width / 2, rep['budget_static_nr_reserve_realized_corr'],
                width, color=NR_COLORS['static'], label='budget-matched static')
    axes[1].bar(x + width / 2, rep['predicted_nr_reserve_realized_corr'],
                width, color=NR_COLORS['predicted'], label='predicted')
    axes[1].set_title('Reserve–actual NR correlation')
    axes[1].set_ylabel('Correlation')
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(rep['rung'])
    axes[1].grid(axis='y', alpha=0.2)

    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, 'fig6_budget_matched.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_variance_interaction(df, agg, metric='slots_included_in_aog', outdir='.'):
    """Fig 6: predicted - static KPI gap vs NR variance scale (Exp. D, tests H5)."""
    sub = agg[agg['exp'] == 'expD'] if 'exp' in agg.columns else agg
    if sub.empty or 'var_scale' not in sub.columns:
        return None
    piv = sub.groupby(['var_scale', 'nr_mode'])[f'{metric}_mean'].mean().unstack('nr_mode')
    if not {'static', 'predicted'}.issubset(piv.columns):
        return None
    gap = piv['predicted'] - piv['static']
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(gap.index, gap.values, marker='o', color='#d62728')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('NR variance scale')
    ax.set_ylabel(f'predicted - static ({metric})')
    ax.set_title('Fig 6 - Value of prediction vs NR variance')
    fig.tight_layout()
    path = os.path.join(outdir, f'fig6_variance_interaction_{metric}.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


# --- Design language for the schematic diagrams (Ocean Dusk palette) -------------
_BG = '#FAFAF7'
_INK = '#264653'          # deep teal text / borders
_OCEAN = ['#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']  # teal, gold, sand, coral


def _card(ax, x, y, w, h, fill, *, edge=_INK, lw=1.4, rounding=0.018):
    """A rounded, softly-shadowed card; returns its centre (cx, cy)."""
    from matplotlib.patches import FancyBboxPatch
    shadow = FancyBboxPatch((x + 0.004, y - 0.006), w, h,
                            boxstyle=f'round,pad=0.004,rounding_size={rounding}',
                            linewidth=0, facecolor='#000000', alpha=0.06, zorder=1)
    ax.add_patch(shadow)
    card = FancyBboxPatch((x, y), w, h,
                          boxstyle=f'round,pad=0.004,rounding_size={rounding}',
                          linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2)
    ax.add_patch(card)
    return x + w / 2.0, y + h / 2.0


def _arrow(ax, p0, p1, color=_INK, lw=2.0, style='-'):
    """Thin arrow with a small filled dot at the source (modern-minimal look)."""
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=14,
                                 linewidth=lw, color=color, linestyle=style,
                                 shrinkA=2, shrinkB=2, zorder=3))
    ax.plot([p0[0]], [p0[1]], marker='o', ms=4, color=color, zorder=4)


def _tint(hex_color, alpha):
    """Blend a hex colour toward the off-white background by `alpha`."""
    bg = (0.980, 0.980, 0.969)
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return tuple(c * alpha + d * (1 - alpha) for c, d in zip((r, g, b), bg))


def _chip(ax, cx, y, text, color, w=0.165, h=0.05):
    """A small filled accent chip with white label, centred horizontally on cx."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((cx - w / 2, y), w, h,
                                boxstyle='round,pad=0.002,rounding_size=0.022',
                                facecolor=color, edgecolor='none', zorder=6))
    ax.text(cx, y + h / 2, text, ha='center', va='center', fontsize=7,
            color='white', weight='bold', zorder=7)


def _pill(ax, x, y, w, h, text, fill, edge):
    """A thin labelled pill (engine sub-component)."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0.002,rounding_size=0.02',
                                facecolor=fill, edgecolor=edge, linewidth=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=7.6,
            color=_INK, zorder=4)


def fig_schematics(outdir='.'):
    """Framework, incumbent-baseline and deregulation-ladder schematics."""
    NRC, SCHC = _OCEAN[3], _OCEAN[0]   # coral = NR module, teal = scheduler module
    written = []

    # ---- Fig 1: the simulator engine with two upgraded modules ---------------------
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    fig.patch.set_facecolor(_BG); ax.set_facecolor(_BG)

    # Central simulation engine.
    ex, ey, ew, eh = 0.40, 0.13, 0.32, 0.74
    _card(ax, ex, ey, ew, eh, _tint(_OCEAN[1], 0.16), edge='#B8902F', lw=1.8)
    ax.text(ex + ew / 2, ey + eh - 0.06, 'anemos',
            ha='center', va='center', fontsize=13, weight='bold', color=_INK)
    ax.text(ex + ew / 2, ey + eh - 0.115, 'discrete-event airline\noperations simulator',
            ha='center', va='center', fontsize=8, color=_INK)
    pills = ['Rotations & tail assignment', 'Maintenance scheduling',
             'Maintenance execution & NR', 'Disruption recovery (MILP)']
    py_top = ey + eh - 0.20
    pw, ph = ew - 0.05, 0.085
    seam_y = {}
    for k, label in enumerate(pills):
        yy = py_top - k * 0.115 - ph
        hl = label in ('Maintenance scheduling', 'Maintenance execution & NR')
        _pill(ax, ex + 0.025, yy, pw, ph,
              label, _tint(_OCEAN[1], 0.42) if hl else '#FFFFFF', '#B8902F')
        seam_y[label] = yy + ph / 2

    # Two "updated module" cards on the left, each feeding a seam.
    # Ordered to match seam height (scheduler -> scheduling pill above,
    # NR -> execution pill below) so the feed arrows run parallel, not crossed.
    schm = _card(ax, 0.02, 0.56, 0.26, 0.27, _tint(SCHC, 0.20), edge=SCHC)
    _chip(ax, schm[0], 0.78, 'updated module', SCHC)
    ax.text(schm[0], 0.70, 'Maintenance scheduling', ha='center', va='center',
            fontsize=9.5, weight='bold', color=_INK)
    ax.text(schm[0], 0.655, 'GNN-SAC policy', ha='center', va='center',
            fontsize=8.2, color=_INK)
    ax.text(schm[0], 0.605, '(upgrades block /\nMILP scheduler)', ha='center', va='center',
            fontsize=7.4, style='italic', color=_INK)

    nrm = _card(ax, 0.02, 0.16, 0.26, 0.27, _tint(NRC, 0.22), edge=NRC)
    _chip(ax, nrm[0], 0.38, 'updated module', NRC)
    ax.text(nrm[0], 0.30, 'Non-routine workload', ha='center', va='center',
            fontsize=9.5, weight='bold', color=_INK)
    ax.text(nrm[0], 0.255, 'task-conditional prediction', ha='center', va='center',
            fontsize=8.2, color=_INK)
    ax.text(nrm[0], 0.205, '(upgrades static\nfleet-average NR)', ha='center', va='center',
            fontsize=7.4, style='italic', color=_INK)

    _arrow(ax, (0.28, 0.63), (ex, seam_y['Maintenance scheduling']), color=SCHC)
    ax.text(0.34, 0.575, 'task $\\rightarrow$ slot\npolicy', ha='center', va='center',
            fontsize=6.8, color=SCHC, zorder=5,
            bbox=dict(boxstyle='round,pad=0.12', fc=_BG, ec='none'))
    _arrow(ax, (0.28, 0.33), (ex, seam_y['Maintenance execution & NR']), color=NRC)
    ax.text(0.34, 0.365, 'per-slot\nNR injection', ha='center', va='center',
            fontsize=6.8, color=NRC, zorder=5,
            bbox=dict(boxstyle='round,pad=0.12', fc=_BG, ec='none'))

    # Output: operational KPIs.
    kpi = _card(ax, 0.74, 0.34, 0.23, 0.32, _tint(_OCEAN[0], 0.16), edge=_INK)
    ax.text(kpi[0], kpi[1] + 0.10, 'Operational KPIs', ha='center', va='center',
            fontsize=9.5, weight='bold', color=_INK)
    ax.text(kpi[0], kpi[1] - 0.02, 'AOG · departure delay\nspillage · drop-out tasks\nuncovered NR labour',
            ha='center', va='center', fontsize=8, color=_INK)
    _arrow(ax, (ex + ew, ey + eh / 2), (0.74, kpi[1]), color='#B8902F')

    p1 = os.path.join(outdir, 'fig1_framework.png')
    fig.savefig(p1, dpi=300, facecolor=_BG, bbox_inches='tight'); plt.close(fig)
    written.append(p1)

    # ---- Incumbent baseline: block cycle plus tentatively scheduled Drop-out tasks -
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    fig.patch.set_facecolor(_BG); ax.set_facecolor(_BG)

    ax.text(0.04, 0.94, 'Incumbent block baseline (R0)', ha='left', va='center',
            fontsize=13, weight='bold', color=_INK)
    ax.text(0.04, 0.885,
            'Predefined A-check blocks repeat over time; routine tasks follow the block,\n'
            'while Drop-out tasks are scheduled tentatively.',
            ha='left', va='top', fontsize=8.5, color=_INK)

    # Repeating block timeline.
    start_x, y, bw, bh, gap = 0.06, 0.56, 0.165, 0.16, 0.045
    blocks = [('Block 1', _OCEAN[0]), ('Block 2', _OCEAN[1]),
              ('Block 3', _OCEAN[2]), ('Block 4', _OCEAN[3])]
    centres = []
    for i, (label, col) in enumerate(blocks):
        x = start_x + i * (bw + gap)
        _card(ax, x, y, bw, bh, _tint(col, 0.28), edge=col, rounding=0.012)
        ax.text(x + bw / 2, y + bh - 0.04, label, ha='center', va='center',
                fontsize=10, weight='bold', color=_INK)
        ax.text(x + bw / 2, y + 0.07, 'scheduled\nA-check', ha='center', va='center',
                fontsize=7.8, color=_INK)
        centres.append((x + bw / 2, y + bh / 2))
        if i < len(blocks) - 1:
            _arrow(ax, (x + bw + 0.004, y + bh / 2),
                   (x + bw + gap - 0.006, y + bh / 2), color='#6B7280', lw=1.4)

    ax.text(0.50, y + bh + 0.05, 'block sequence repeats at predetermined intervals',
            ha='center', va='bottom', fontsize=7.8, color='#6B7280')

    # In-block task lane.
    ax.text(0.055, 0.44, 'In-block tasks', ha='left', va='center',
            fontsize=9, weight='bold', color=_INK)
    for i, (cx, _) in enumerate(centres):
        tx = cx - 0.065
        for j, task in enumerate([f'T{i + 1}a', f'T{i + 1}b', f'T{i + 1}c']):
            _pill(ax, tx + j * 0.045, 0.38, 0.04, 0.045, task, '#FFFFFF', '#9CA3AF')
        _arrow(ax, (cx, 0.425), (cx, y + 0.01), color='#9CA3AF', lw=1.0)

    # Drop-out lane with tentative assignment.
    ax.text(0.055, 0.26, 'Drop-out tasks', ha='left', va='center',
            fontsize=9, weight='bold', color=_INK)
    for x0, x1, block_c, label in [(0.24, 0.41, centres[1][0], 'DO-1'),
                                   (0.58, 0.75, centres[2][0], 'DO-2')]:
        ax.add_patch(Rectangle((x0, 0.205), x1 - x0, 0.065,
                               facecolor=_tint(_OCEAN[3], 0.18),
                               edgecolor=_OCEAN[3], linewidth=1.0,
                               linestyle=':', zorder=1))
        ax.text((x0 + x1) / 2, 0.237, 'tentative\nschedule',
                ha='center', va='center', fontsize=7.2, color=_OCEAN[3])
        ax.scatter([(x0 + x1) / 2], [0.15], s=36, color=_OCEAN[3], zorder=5)
        ax.text((x0 + x1) / 2, 0.105, label, ha='center', va='top',
                fontsize=7.5, color=_OCEAN[3])
        _arrow(ax, ((x0 + x1) / 2, 0.27), (block_c, y - 0.005),
               color=_OCEAN[3], lw=1.7)

    p0 = os.path.join(outdir, 'fig_incumbent_baseline.pdf')
    fig.savefig(p0, dpi=300, facecolor=_BG, bbox_inches='tight'); plt.close(fig)
    written.append(p0)

    # ---- Fig 2: deregulation ladder ------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    fig.patch.set_facecolor(_BG); ax.set_facecolor(_BG)

    rungs = [('R0', 'strict', 'Block baseline\nfixed incumbent\nA-check blocks', _OCEAN[0]),
             ('R1', 'moderate', 'GNN-SAC policy\n~4 slots / aircraft / year\ncurrent-practice constraint', _OCEAN[1]),
             ('R2', 'free', 'GNN-SAC policy\n~12 hangar slots / year\nhigher-capacity constraint', _OCEAN[3])]
    w, h = 0.27, 0.30
    xs = [0.04, 0.37, 0.70]
    bases = [0.16, 0.34, 0.52]   # gentle ascending staircase
    centres = [(xs[i] + w / 2, bases[i] + h / 2) for i in range(3)]

    # Guide ramp behind the cards (sense of rising freedom).
    ax.plot([centres[0][0], centres[2][0]], [centres[0][1], centres[2][1]],
            color='#C9C9C2', lw=10, alpha=0.5, solid_capstyle='round', zorder=0)

    edges = ['#B8902F' if r[0] == 'R1' else r[3] for r in rungs]
    for i, (rid, name, knob, col) in enumerate(rungs):
        cx, cyc = _card(ax, xs[i], bases[i], w, h, _tint(col, 0.28), edge=edges[i])
        _chip(ax, xs[i] + 0.05, bases[i] + h - 0.055, rid, edges[i], w=0.075, h=0.05)
        ax.text(xs[i] + w - 0.02, bases[i] + h - 0.03, name, ha='right', va='center',
                fontsize=10, weight='bold', style='italic', color=_INK)
        ax.text(cx, cyc - 0.035, knob, ha='center', va='center', fontsize=8.2, color=_INK)

    transitions = ['replace fixed\nblock policy', 'relax slot-frequency\nconstraint']
    for i in range(2):
        _arrow(ax, (xs[i] + w, bases[i] + h / 2),
               (xs[i + 1], bases[i + 1] + h / 2), color='#6B7280', lw=1.8)
        # Right-aligned just left of the next card so labels never tuck behind it.
        ax.text(xs[i + 1] - 0.01, bases[i + 1] + h / 2 + 0.055, transitions[i],
                ha='right', va='bottom', fontsize=7, color='#6B7280')

    ax.annotate('', xy=(0.99, 0.06), xytext=(0.02, 0.06),
                arrowprops=dict(arrowstyle='-|>', color=_INK, lw=1.4))
    ax.text(0.5, 0.01, 'increasing optimiser freedom  ·  from industry-practice slots to hangar-capacity slots',
            ha='center', va='bottom', fontsize=8.5, style='italic', color=_INK)

    p2 = os.path.join(outdir, 'fig2_ladder.png')
    fig.savefig(p2, dpi=300, facecolor=_BG, bbox_inches='tight'); plt.close(fig)
    written.append(p2)
    return written


def make_all(scenario='default_run', output_dir=None, run_suffix=None, overview_csv=None,
             budget_csv=None):
    outdir = _outdir(output_dir)
    if overview_csv:
        df = pd.read_csv(overview_csv)
    else:
        df = load_overviews(scenario, output_dir, run_suffix=run_suffix)
    df = _with_derived_metrics(df) if not df.empty else df
    agg = aggregate(df) if not df.empty else df
    written = []
    written += fig_schematics(outdir)
    if not agg.empty:
        for path in [fig_optimizer_panel(df, agg, outdir=outdir),
                     fig_buffer_pareto(df, agg, outdir=outdir),
                     fig_robustness(df, agg, outdir=outdir),
                     fig_budget_matched(budget_csv, outdir=outdir)]:
            if path:
                written.append(path)
    else:
        print('No experiment outputs found; rendered schematic figures only.')
    print('Wrote figures:')
    for p in written:
        print('  ', p)
    return written


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='default_run')
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--run-suffix', default=None)
    ap.add_argument('--overview-csv', default=None,
                    help='Use an analysis-ready overview CSV instead of scanning Data/output folders.')
    ap.add_argument('--budget-csv', default=None,
                    help='Use an Experiment F budget-matched report CSV for the budget figure.')
    args = ap.parse_args()
    make_all(args.scenario, args.output_dir, args.run_suffix, args.overview_csv, args.budget_csv)
