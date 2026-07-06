from config import G
import pyomo.environ as pyo


def make_solver(name, mipgap=None):
    solver = pyo.SolverFactory(name)
    if mipgap is not None:
        solver.options['mipgap'] = mipgap
    if G.SOLVER_THREADS is not None and _supports_threads(name):
        solver.options['threads'] = G.SOLVER_THREADS
    return solver


def _supports_threads(name):
    return isinstance(name, str) and name.startswith('gurobi')
