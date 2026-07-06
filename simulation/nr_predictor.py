"""
Conditional Non-Routine (NR) predictor -- inference-only shim for the simulator.

When `G.NR_MODE == 'predicted'`, the simulator replaces the
fleet-average *sampled* NR value with a *conditional* magnitude produced here from the
vendored offline artifact (Data/input/nr_prediction/). This module is deliberately
self-contained and pandas-only: no sklearn, no model objects, no training data at sim
runtime. The artifact is produced by generate_mock_nr_artifact.py.

Mass-preservation contract (PAPER_DESIGN sec. 4.4): the *gate* stays identical to the
static model (anemos draws `random.uniform(0,1) > fleet probability_NR` per routine task).
Only the injected *magnitude* changes, and it is built to preserve the static model's per-
occurrence NR mass while moving its *placement*:

    magnitude(task) = q(code, Q)        # anemos calls magnitude_for_codes(fleet, [code]) per task
    q(code, Q) = quantile(code, Q)

anemos gates and injects NR PER routine task (see anemos.__get_NR_for_slot) and sums the draws
over the slot, so total NR scales with the work in the slot. Each occurrence's magnitude tracks
the calibrated centre (q50 ~= static mean per occurrence). A risk-buffer quantile Q>0.5 raises
the value above the centre on purpose -- that extra mass is the cost of the buffer studied in
Exp. B. (magnitude_for_codes still accepts a list and averages it, so a multi-code call remains
valid, but anemos now passes one code at a time.) Workload dispersion
(variance_scale) is modelled separately, on the *realized* NR draw in anemos, not on this
reserved estimate (EXPERIMENT_PLAN sec. 2.3); it is recorded here only for traceability.
"""

import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd

FLEET_DEFAULT_CODE = '__fleet_default__'


def _integrate_trapezoid(values, levels):
    """Compatibility wrapper for NumPy 1.x/2.x trapezoid integration."""
    if hasattr(np, 'trapezoid'):
        return float(np.trapezoid(values, levels))
    return float(np.trapz(values, levels))


def artifact_paths():
    """Resolve the (conditional_csv, calibration_json) absolute paths from config."""
    from config import directories, G
    base = directories.nr_prediction
    return (os.path.join(base, G.NR_CONDITIONAL_FILE),
            os.path.join(base, G.NR_CALIBRATION_FILE))


class NRPredictor:
    """Loads the NR artifact once and answers per-slot magnitude queries.

    One instance per simulation iteration (holds running totals for the calibration
    report). Thread/process-safe by construction: it is read-only after __init__ apart
    from the lightweight accumulators, and anemos runs one predictor per process.
    """

    def __init__(self, conditional_csv=None, calibration_json=None,
                 buffer_quantile=0.5, variance_scale=1.0, calibrate_to_static=True):
        if conditional_csv is None or calibration_json is None:
            conditional_csv, calibration_json = artifact_paths()
        self.buffer_quantile = float(buffer_quantile)
        self.variance_scale = float(variance_scale)
        self.calibrate_to_static = bool(calibrate_to_static)

        with open(calibration_json, 'r') as f:
            self.calibration = json.load(f)
        self.q_cols = self.calibration['quantile_columns']
        self.q_levels = np.asarray(self.calibration['quantile_levels'], dtype=float)
        self.max_duration_nr = float(self.calibration.get('max_duration_nr', 50.0))
        self.fleet_info = self.calibration.get('fleets', {})

        df = pd.read_csv(conditional_csv)
        # Lookup: (fleet, code) -> (qvalues ndarray, p_nr); plus per-fleet default row.
        self._lookup = {}
        self._fleet_default = {}
        self._realized_scales = {}
        for _, row in df.iterrows():
            fleet = str(row['fleet'])
            code = str(row['rt_code'])
            qvals = np.asarray([float(row[c]) for c in self.q_cols], dtype=float)
            entry = (qvals, float(row.get('p_nr', np.nan)))
            if code == FLEET_DEFAULT_CODE:
                self._fleet_default[fleet] = entry
            else:
                self._lookup[(fleet, code)] = entry

        # Calibration accumulators, per fleet, for the post-run report.
        self._acc = defaultdict(lambda: {'sum_pred': 0.0, 'n_inject': 0,
                                          'codes_hit': 0, 'codes_miss': 0, 'slots_no_code': 0,
                                          'sum_prob': 0.0, 'n_prob': 0})

    # ---- core lookup -------------------------------------------------------------
    def _q_value(self, qvals):
        """Reserved-estimate value at the buffer quantile. Workload dispersion is applied to the
        realized NR draw in anemos (not here), so variance_scale no longer modifies this estimate
        -- Framing 1, EXPERIMENT_PLAN sec. 2.3."""
        return float(np.interp(self.buffer_quantile, self.q_levels, qvals))

    def _runtime_scale(self, fleet):
        if not self.calibrate_to_static:
            return 1.0
        return float(self.fleet_info.get(fleet, {}).get('runtime_scale', 1.0))

    def _realized_runtime_scale(self, fleet):
        """Scale inverse-CDF execution draws to the static mean per occurrence.

        The exported artifact calibrates q50 to the static mean, which is appropriate for the
        planning reserve.  After execution was changed from the deterministic q50 to a stochastic
        inverse-CDF draw, applying that same scale made realized mass too large because the
        right-skewed conditional distribution has E[X] > q50.  Use the fleet-default conditional
        distribution to recover E[X] and add a second scale only for realized draws.
        """
        if fleet in self._realized_scales:
            return self._realized_scales[fleet]
        planning_scale = self._runtime_scale(fleet)
        if not self.calibrate_to_static:
            return planning_scale
        info = self.fleet_info.get(fleet, {})
        static_mean = info.get('static_mean_per_occurrence')
        default = self._fleet_default.get(fleet)
        if not static_mean or default is None:
            return planning_scale
        qvals = default[0]
        # np.interp uses constant tails outside the exported quantile range. Integrating the
        # corresponding piecewise-linear inverse CDF gives its expected execution magnitude.
        levels = np.concatenate(([0.0], self.q_levels, [1.0]))
        values = np.concatenate(([qvals[0]], qvals, [qvals[-1]]))
        expected_raw = _integrate_trapezoid(values, levels)
        if expected_raw <= 0:
            return planning_scale
        return float(static_mean) / expected_raw

    def calibrate_realized_code_mix(self, fleet, task_codes):
        """Calibrate execution mass to the task-code mix actually present in the run."""
        fleet = str(fleet)
        codes = list(task_codes)
        info = self.fleet_info.get(fleet, {})
        static_mean = info.get('static_mean_per_occurrence')
        if not self.calibrate_to_static or not static_mean or not codes:
            return
        levels = np.concatenate(([0.0], self.q_levels, [1.0]))
        expected = []
        for code in codes:
            entry = self._code_entry(fleet, code)
            if entry is None:
                continue
            qvals = entry[0]
            values = np.concatenate(([qvals[0]], qvals, [qvals[-1]]))
            expected.append(_integrate_trapezoid(values, levels))
        if expected and np.mean(expected) > 0:
            self._realized_scales[fleet] = float(static_mean) / float(np.mean(expected))

    def _code_entry(self, fleet, code):
        """(qvalues ndarray) for a (fleet, code), falling back to the per-fleet default; None if unknown."""
        fleet = str(fleet)
        entry = self._lookup.get((fleet, str(code)))
        if entry is None:
            entry = self._fleet_default.get(fleet)
        return entry

    def reserved_magnitude(self, fleet, code):
        """PLANNING reserve for one routine task: the code's conditional NR magnitude at the buffer
        quantile (higher NR_BUFFER_QUANTILE -> larger reserve). Pure: no RNG, no accumulators."""
        entry = self._code_entry(fleet, code)
        if entry is None:
            return 0.0
        val = float(np.interp(self.buffer_quantile, self.q_levels, entry[0])) * self._runtime_scale(str(fleet))
        return max(0.0, min(val, self.max_duration_nr))

    def realized_magnitude(self, fleet, code, u):
        """EXECUTION draw for one routine task: inverse-CDF sample of the code's conditional NR
        magnitude at uniform draw ``u`` (caller supplies the seeded RNG value). Updates the
        calibration accumulators so the post-run report reflects realized NR."""
        fleet = str(fleet)
        acc = self._acc[fleet]
        entry = self._lookup.get((fleet, str(code)))
        if entry is None:
            entry = self._fleet_default.get(fleet)
            acc['codes_miss'] += 1
        else:
            acc['codes_hit'] += 1
        if entry is None:
            return 0.0
        val = float(np.interp(u, self.q_levels, entry[0])) * self._realized_runtime_scale(fleet)
        val = round(max(0.0, min(val, self.max_duration_nr)), 2)
        acc['sum_pred'] += val
        acc['n_inject'] += 1
        return val

    def has_fleet(self, fleet):
        """True if the artifact carries data for this fleet (per-code rows or a default).
        When False, callers fall back to the static NR distribution instead of injecting 0."""
        fleet = str(fleet)
        return fleet in self.fleet_info or fleet in self._fleet_default

    def slot_probability(self, fleet, task_codes):
        """Per-slot NR *probability* (the gate) implied by the prediction: the mean per-code
        p_nr over the slot's routine tasks, scaled so the fleet-average equals the static
        probability_NR (expected occurrence preserved; the prediction redistributes risk).
        Returns None if the fleet is unknown (caller falls back to the static gate)."""
        fleet = str(fleet)
        info = self.fleet_info.get(fleet)
        if info is None or info.get('probability_NR') is None:
            return None
        base = float(info['probability_NR'])
        prob_scale = float(info.get('prob_scale', 1.0))
        pvals = []
        for code in task_codes:
            entry = self._lookup.get((fleet, str(code)))
            if entry is not None and not np.isnan(entry[1]):
                pvals.append(entry[1])
        p = base if not pvals else float(np.mean(pvals)) * prob_scale
        p = min(max(p, 0.0), 1.0)
        acc = self._acc[fleet]
        acc['sum_prob'] += p
        acc['n_prob'] += 1
        return p

    def magnitude_for_codes(self, fleet, task_codes):
        """Predicted NR labour hours for a hangar slot, given its routine-task codes.
        Returns 0.0 only if the fleet is entirely unknown (logged)."""
        fleet = str(fleet)
        acc = self._acc[fleet]
        values = []
        for code in task_codes:
            entry = self._lookup.get((fleet, str(code)))
            if entry is None:
                entry = self._fleet_default.get(fleet)
                acc['codes_miss'] += 1
            else:
                acc['codes_hit'] += 1
            if entry is not None:
                values.append(self._q_value(entry[0]))

        if not values:
            # No routine-task codes resolved -> fall back to the fleet-default magnitude.
            default = self._fleet_default.get(fleet)
            if default is None:
                return 0.0
            acc['slots_no_code'] += 1
            values = [self._q_value(default[0])]

        magnitude = float(np.mean(values)) * self._runtime_scale(fleet)
        magnitude = max(0.0, min(magnitude, self.max_duration_nr))
        magnitude = round(magnitude, 2)
        acc['sum_pred'] += magnitude
        acc['n_inject'] += 1
        return magnitude

    # ---- reporting ---------------------------------------------------------------
    def calibration_report(self):
        """Per-fleet realized predicted NR vs the static expected NR per slot
        (PAPER_DESIGN sec. 4.4 mass-preservation check). Returns a DataFrame."""
        rows = []
        for fleet, acc in self._acc.items():
            n = max(acc['n_inject'], 1)
            mean_pred = acc['sum_pred'] / n
            static_mean = self.fleet_info.get(fleet, {}).get('static_mean_per_occurrence', np.nan)
            static_prob = self.fleet_info.get(fleet, {}).get('probability_NR', np.nan)
            total_codes = acc['codes_hit'] + acc['codes_miss']
            n_prob = max(acc['n_prob'], 1)
            mean_prob = acc['sum_prob'] / n_prob if acc['n_prob'] else np.nan
            rows.append({
                'fleet': fleet,
                'n_injections': acc['n_inject'],
                'mean_predicted_per_occurrence': round(mean_pred, 4),
                'static_mean_per_occurrence': static_mean,
                'mass_ratio_pred_over_static': (round(mean_pred / static_mean, 4)
                                                if static_mean else np.nan),
                'n_prob_evals': acc['n_prob'],
                'mean_predicted_probability': (round(mean_prob, 4) if acc['n_prob'] else np.nan),
                'static_probability': static_prob,
                'prob_ratio_pred_over_static': (round(mean_prob / static_prob, 4)
                                                if acc['n_prob'] and static_prob else np.nan),
                'code_coverage': (round(acc['codes_hit'] / total_codes, 4)
                                  if total_codes else np.nan),
                'buffer_quantile': self.buffer_quantile,
                'variance_scale': self.variance_scale,
            })
        return pd.DataFrame(rows)


def load_predictor():
    """Convenience constructor pulling all knobs from config (G.NR_*)."""
    from config import G
    csv_path, json_path = artifact_paths()
    return NRPredictor(conditional_csv=csv_path, calibration_json=json_path,
                       buffer_quantile=G.NR_BUFFER_QUANTILE,
                       variance_scale=G.NR_VARIANCE_SCALE,
                       calibrate_to_static=G.NR_CALIBRATE_TO_STATIC)
