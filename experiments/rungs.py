"""
Deregulation ladder (EXPERIMENT_PLAN.md sec. 2.1).

Each rung is a set of ANEMOS_* environment overrides (read by config.py at import in a
fresh subprocess). The ladder goes from the incumbent block schedule to the free optimiser:

    R0 strict   : block-based baseline (MaintenanceSchedulerBaselineBlock) -- the incumbent
                  fixed A-check block practice, no optimisation.
    R1 moderate : SAC-GNN optimiser, constrained -- wide 60-day slot-combine window (scenario 1)
                  consolidates maintenance into few, full slots (fewest opportunities).
    R2 free     : SAC-GNN optimiser, free -- narrow 7-day slot-combine window (scenario 2)
                  spreads maintenance across many frequent slots (the most opportunities).

The relaxation axis is the SAC-GNN slot-combine window: a NARROWER window leaves slots un-merged, so
there are more, more-frequent maintenance opportunities (freer). This is not a labour cap -- the block
baseline and SAC-GNN do not enforce the MILP per-slot labour constraint, which is no longer used.
NR mode is the crossed second factor (static vs predicted).
"""

RUNGS = {
    'R0': {
        'label': 'R0_strict',
        'description': 'Strict: block-based baseline with 75-day due-proximity gate (incumbent fixed A-check blocks, no optimisation)',
        'env': {
            'ANEMOS_MAINTENANCE_SCHEDULE': '5',          # baseline MRI block scheduler
            'ANEMOS_BLOCK_DUE_GATE_DAYS': '75',           # calibrated R0: preserves block system, avoids unrealistic early execution
        },
    },
    'R1': {
        'label': 'R1_moderate',
        'description': 'Moderate: SAC-GNN optimiser, constrained -- wide 60-day combine window, maintenance consolidated into few slots',
        'env': {
            'ANEMOS_MAINTENANCE_SCHEDULE': '6',          # SAC-GNN scenario 1 (60-day combine window)
        },
    },
    'R2': {
        'label': 'R2_free',
        'description': 'Free: SAC-GNN optimiser, free -- narrow 7-day combine window, maintenance spread across many frequent slots',
        'env': {
            'ANEMOS_MAINTENANCE_SCHEDULE': '7',          # SAC-GNN scenario 2 (7-day combine window)
        },
    },
}

RUNG_ORDER = ['R0', 'R1', 'R2']

NR_MODES = {
    # Incumbent allocation: reserve the mean total realized NR hours from the previous five
    # completed A-check work packages for the same fleet/check type.
    'static':    {'ANEMOS_NR_MODE': 'static'},
    # Keep the same fleet-level occurrence gate as the static baseline and replace
    # only the realized NR magnitude with the conditional estimate. This preserves
    # paired occurrence draws and makes NR_BUFFER_QUANTILE effective in Exp. B.
    'predicted': {
        'ANEMOS_NR_MODE': 'predicted',
        'ANEMOS_NR_PREDICTED_SIGNAL': 'magnitude',
    },
}


def rung_env(rung_key):
    """Return a copy of the env-override dict for a rung key ('R0'/'R1'/'R2')."""
    return dict(RUNGS[rung_key]['env'])


def nr_env(nr_mode, buffer_quantile=None, variance_scale=None):
    """Env overrides for an NR mode, optionally with buffer quantile / variance scale."""
    env = dict(NR_MODES[nr_mode])
    if buffer_quantile is not None:
        env['ANEMOS_NR_BUFFER_QUANTILE'] = str(buffer_quantile)
    if variance_scale is not None:
        env['ANEMOS_NR_VARIANCE_SCALE'] = str(variance_scale)
    return env
