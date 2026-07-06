"""
PAPER_DESIGN.md batch experiment runner (Phase 4).

Sweeps the crossed factors of the paper by launching one *subprocess per experiment cell*
(``python main.py``), passing the cell's configuration through ANEMOS_* environment variables
that config.py applies at import. Each cell writes a deterministically-named output folder
(``<scenario>_<tag>``) under Data/output/ which experiments/analysis.py then aggregates.

Experiments (see EXPERIMENT_PLAN.md):
  A  integrated policy evaluation           -> optimiser/prediction value across R0/R1/R2
  B  NR-buffer decision support              -> capacity-coverage frontier at a fixed rung
  D  policy robustness to NR uncertainty     -> variance sweep by rung and NR mode
  F  budget-matched NR reservation           -> static reserve scaled to predicted budget

C/E remain available for legacy sensitivity work but are outside the minimum paper design.

Usage (run from repo root, in the full sim environment with simpy/pyomo installed):
    python -m experiments.run_experiments --exp A --iterations 20 --duration 365
    # Append iterations 5-9 to deterministic folders from an earlier five-iteration campaign.
    python -m experiments.run_experiments --exp A --iterations 5 --iteration-start 5 \
        --tag-suffix _updated5_20260625
    # Rerun one policy rung after a rung-definition change.
    python -m experiments.run_experiments --exp A --rung R0 --iterations 5
    # Low/high uncertainty extension at one policy rung.
    python -m experiments.run_experiments --exp D --rung R1 --variance-scales 0.5,2.0
    python -m experiments.run_experiments --exp all --dry-run        # print the plan only
    python -m experiments.run_experiments --exp B --rung R1 --python py
    # many-core box: run all cells concurrently, 10 iteration-cores each, threads pinned.
    # Size it so max_parallel_cells * cores_per_cell ~= physical cores (e.g. 19*10=190 on 256).
    python -m experiments.run_experiments --exp all --max-parallel-cells 19 --cores-per-cell 10
    # Budget-matched static reserve, all three rungs concurrently.
    python -m experiments.run_experiments --exp F --iterations 5 --max-parallel-cells 3
"""

import os
import sys
import time
import math
import argparse
import subprocess
from itertools import product

# BLAS/torch/solver thread-pinning: when many cells run concurrently, each sim must stay
# single-threaded or the processes oversubscribe the cores and slow each other down. The
# cell subprocesses inherit these (see _cell_env).
_THREAD_PIN_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS')

from experiments.rungs import RUNGS, RUNG_ORDER, rung_env, nr_env

NR_MODE_LIST = ['static', 'predicted']
# Default sweep grids (override on the CLI as needed).
BUFFER_QUANTILES = [0.5, 0.6, 0.75, 0.9]
VARIANCE_SCALES = [0.5, 1.0, 1.5, 2.0]
RESERVES_LEVELS = [0, 1, 2]
STATIC_RESERVE_SCALES_EXP_F = {
    'R0': 1.28,
    'R1': 1.28,
    'R2': 1.34,
}


# --------------------------------------------------------------------------- cells
def _cell(exp, name, env):
    """A single experiment cell: an output tag + the env overrides that define it."""
    return {'exp': exp, 'tag': f'{exp}_{name}', 'env': dict(env)}


def _exp_f_scale(rung):
    key = f'ANEMOS_EXP_F_STATIC_RESERVE_SCALE_{rung}'
    return float(os.environ.get(key, STATIC_RESERVE_SCALES_EXP_F[rung]))


def cells_expA(rung=None):
    cells = []
    rungs = [rung] if rung else RUNG_ORDER
    for rung_key, nr in product(rungs, NR_MODE_LIST):
        env = {**rung_env(rung_key), **nr_env(nr, buffer_quantile=0.5)}
        cells.append(_cell('expA', f'{rung_key}_{nr}', env))
    return cells


def cells_expB(rung='R2'):
    cells = []
    # Static baseline (buffer irrelevant) + predicted across buffer quantiles.
    cells.append(_cell('expB', f'{rung}_static', {**rung_env(rung), **nr_env('static')}))
    for q in BUFFER_QUANTILES:
        env = {**rung_env(rung), **nr_env('predicted', buffer_quantile=q)}
        cells.append(_cell('expB', f'{rung}_pred_q{q}', env))
    return cells


def cells_expC():
    cells = []
    # Robustness at R2: optimiser-alone (static) vs full system (predicted) under NR severity.
    for nr, v, res in product(NR_MODE_LIST, VARIANCE_SCALES, [0, 1]):
        env = {**rung_env('R2'), **nr_env(nr, buffer_quantile=0.5, variance_scale=v)}
        env['ANEMOS_RESERVES_PER_DAY'] = str(res)
        cells.append(_cell('expC', f'{nr}_v{v}_res{res}', env))
    return cells


def cells_expD(rung='R2', variance_scales=None):
    cells = []
    scales = VARIANCE_SCALES if variance_scales is None else variance_scales
    for nr, v in product(NR_MODE_LIST, scales):
        env = {**rung_env(rung), **nr_env(nr, buffer_quantile=0.5, variance_scale=v)}
        cells.append(_cell('expD', f'{rung}_{nr}_v{v}', env))
    return cells


def cells_expE(rung='R0'):
    cells = []
    for nr, res in product(NR_MODE_LIST, RESERVES_LEVELS):
        env = {**rung_env(rung), **nr_env(nr, buffer_quantile=0.5)}
        env['ANEMOS_RESERVES_PER_DAY'] = str(res)
        cells.append(_cell('expE', f'{rung}_{nr}_res{res}', env))
    return cells


def cells_expF(rung=None):
    cells = []
    rungs = [rung] if rung else RUNG_ORDER
    for rung_key in rungs:
        env = {**rung_env(rung_key), **nr_env('static', buffer_quantile=0.5)}
        env['ANEMOS_NR_STATIC_RESERVE_SCALE'] = str(_exp_f_scale(rung_key))
        cells.append(_cell('expF', f'{rung_key}_static_budgetmatched', env))
    return cells


EXPERIMENTS = {'A': cells_expA, 'B': cells_expB, 'C': cells_expC, 'D': cells_expD, 'E': cells_expE,
               'F': cells_expF}


# --------------------------------------------------------------------------- run
def _cell_env(cell, scenario, iterations, duration, base_seed, run_mode, cores_per_cell,
              iteration_start=0):
    env = os.environ.copy()
    env.update(cell['env'])
    env['ANEMOS_SCENARIO'] = scenario
    env['ANEMOS_RUN_TAG'] = cell['tag']
    env['ANEMOS_SIM_ITERATIONS'] = str(iterations)
    env['ANEMOS_SIM_DURATION'] = str(duration)
    env['ANEMOS_RUN_MODE'] = str(run_mode)
    env['ANEMOS_BASE_SEED'] = str(base_seed)
    env['ANEMOS_ITERATION_START'] = str(iteration_start)
    if cores_per_cell is not None:
        env['ANEMOS_PARALLEL_RUN_N_CORES'] = str(cores_per_cell)
    # Cap each optimisation solve and SAC inference worker when cells/iterations run
    # concurrently. Without these caps Gurobi may use dozens of threads per process.
    env.setdefault('ANEMOS_SOLVER_THREADS', '1')
    env.setdefault('ANEMOS_SAC_GNN_CPU_THREADS', '1')
    for var in _THREAD_PIN_VARS:   # keep each sim single-threaded; see _THREAD_PIN_VARS
        env[var] = '1'
    return env


def _print_cell(cell, scenario):
    out_folder = f'{scenario}_{cell["tag"]}'
    overrides = ' '.join(f'{k}={v}' for k, v in cell['env'].items())
    print(f"[{cell['exp']}] {out_folder}\n      {overrides}")


def run_cells(cells, scenario, iterations, duration, base_seed, run_mode, cores_per_cell,
              max_parallel_cells, python_exe, dry_run, iteration_start=0):
    """Launch cells, up to max_parallel_cells subprocesses at a time. Returns failure count."""
    for cell in cells:
        _print_cell(cell, scenario)
    if dry_run:
        return 0

    failures = 0
    running = []   # list of (Popen, cell)
    pending = list(cells)
    while pending or running:
        while pending and len(running) < max_parallel_cells:
            cell = pending.pop(0)
            env = _cell_env(cell, scenario, iterations, duration, base_seed, run_mode,
                            cores_per_cell, iteration_start)
            proc = subprocess.Popen([python_exe, 'main.py'], env=env, cwd=os.getcwd())
            running.append((proc, cell))
        still = []
        for proc, cell in running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, cell))
            elif rc != 0:
                failures += 1
                print(f"  !! cell failed (exit {rc}): {scenario}_{cell['tag']}", file=sys.stderr)
        running = still
        if running and len(running) >= max_parallel_cells or (running and not pending):
            time.sleep(0.5)
    return failures


def time_one(cells, scenario, duration, base_seed, cores_per_cell, max_parallel_cells,
             iterations, python_exe):
    """Run one iteration of the most expensive (SAC-GNN) cell, then project the full grid."""
    probe = next((c for c in cells if c['env'].get('ANEMOS_MAINTENANCE_SCHEDULE') in ('6', '7')),
                 cells[0])
    sched = probe['env'].get('ANEMOS_MAINTENANCE_SCHEDULE', '?')
    print(f'Timing 1 iteration of {scenario}_{probe["tag"]} (sched {sched}) at {duration}d ...')
    env = _cell_env(probe, scenario, iterations=1, duration=duration, base_seed=base_seed,
                    run_mode=0, cores_per_cell=1, iteration_start=0)
    t0 = time.time()
    rc = subprocess.run([python_exe, 'main.py'], env=env, cwd=os.getcwd()).returncode
    t = time.time() - t0
    if rc != 0:
        print(f'  !! probe cell failed (exit {rc}); cannot project.', file=sys.stderr)
        return 1

    cpc = cores_per_cell or iterations                   # recommended: cores_per_cell == iterations
    cell_waves = math.ceil(len(cells) / max_parallel_cells)
    iter_waves = math.ceil(iterations / cpc)
    est = cell_waves * iter_waves * t
    print(f'\n  one iteration (incl. import + data load): {t:.1f} s')
    print(f'  grid: {len(cells)} cells x {iterations} iters; '
          f'{max_parallel_cells} cells concurrent x {cpc} cores/cell')
    print(f'  cell waves = ceil({len(cells)}/{max_parallel_cells}) = {cell_waves};  '
          f'iter waves = ceil({iterations}/{cpc}) = {iter_waves}')
    print(f'\nProjected grid wall-clock ~= {cell_waves} x {iter_waves} x {t:.1f}s '
          f'= {est/60:.1f} min ({est/3600:.2f} h)')
    print('  (upper bound: the probe is the most expensive SAC-GNN cell, and block/R0 cells are '
          'cheaper; each iter-wave is charged its own import+load overhead.)')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description='PAPER_DESIGN.md batch experiment runner')
    ap.add_argument('--exp', default='A', help="Experiment(s): A/B/C/D/E/F or 'all'")
    ap.add_argument('--scenario', default='default_run', help='Scenario id (Scenarios_simulation)')
    ap.add_argument('--iterations', type=int, default=10, help='Stochastic iterations per cell')
    ap.add_argument('--duration', type=int, default=365, help='Sim duration [days] (1-year window)')
    ap.add_argument('--base-seed', type=int, default=1000, help='Paired base seed (cells share it)')
    ap.add_argument('--iteration-start', type=int, default=0,
                    help='First simulation-iteration index. Use 5 with --iterations 5 to append '
                         'iterations 5-9 to an existing deterministic run folder.')
    ap.add_argument('--tag-suffix', default='',
                    help='Suffix appended to every cell tag, e.g. _updated5_20260625.')
    ap.add_argument('--run-mode', type=int, default=1000, choices=[0, 1000],
                    help='0 = series, 1000 = parallel iterations within a cell (default)')
    ap.add_argument('--max-parallel-cells', type=int, default=1,
                    help='Cells to run concurrently (default 1 = sequential). On a many-core box set '
                         'this so max_parallel_cells * cores_per_cell ~= available cores.')
    ap.add_argument('--cores-per-cell', type=int, default=None,
                    help='Iteration parallelism within each cell (ANEMOS_PARALLEL_RUN_N_CORES). '
                         'Default leaves the config value (8). Capped usefully at --iterations.')
    ap.add_argument('--rung', default=None,
                    help='Fixed rung for A/B/D/F (B/D default R2) and E (default R0)')
    ap.add_argument('--variance-scales', default=None,
                    help='Comma-separated variance subset for D, e.g. 0.5,2.0.')
    ap.add_argument('--python', default=sys.executable, help='Python executable for subprocesses')
    ap.add_argument('--dry-run', action='store_true', help='Print the plan without running')
    ap.add_argument('--time-one', action='store_true',
                    help='Run one iteration of the most expensive (SAC-GNN) cell at --duration and '
                         'project the full grid wall-clock. Run this once before the full sweep.')
    args = ap.parse_args(argv)

    exps = list('ABCDEF') if args.exp.lower() == 'all' else [args.exp.upper()]
    cells = []
    variance_scales = (
        [float(v.strip()) for v in args.variance_scales.split(',') if v.strip()]
        if args.variance_scales else None
    )
    for e in exps:
        builder = EXPERIMENTS[e]
        if e == 'A' and args.rung:
            cells += builder(args.rung)
        elif e == 'D':
            cells += builder(args.rung or 'R2', variance_scales=variance_scales)
        elif e == 'B' and args.rung:
            cells += builder(args.rung)
        elif e == 'E' and args.rung:
            cells += builder(args.rung)
        elif e == 'F' and args.rung:
            cells += builder(args.rung)
        else:
            cells += builder()
    if args.tag_suffix:
        for cell in cells:
            cell['tag'] += args.tag_suffix

    if args.time_one:
        return time_one(cells, args.scenario, args.duration, args.base_seed, args.cores_per_cell,
                        args.max_parallel_cells, args.iterations, args.python)

    print(f'Planning {len(cells)} cell(s) across experiment(s) {exps}; '
          f'{args.iterations} iter x {args.duration}d each; '
          f'{args.max_parallel_cells} cell(s) concurrent, '
          f'{args.cores_per_cell or "config"} core(s)/cell. dry_run={args.dry_run}\n')
    failures = run_cells(cells, args.scenario, args.iterations, args.duration, args.base_seed,
                         args.run_mode, args.cores_per_cell, args.max_parallel_cells,
                         args.python, args.dry_run, args.iteration_start)
    print(f'\nDone. {len(cells)} cell(s), {failures} failure(s).')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
