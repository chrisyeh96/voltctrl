"""Convex body chasing base class + utilities."""
from __future__ import annotations

import io
import os
import pickle
from typing import Any

import cvxpy as cp
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def solve_prob(prob: cp.Problem, log: tqdm | io.TextIOBase | None = None,
               name: str = '', indent: str = '') -> None:
        if len(indent) > 0:
             name = f'{indent} {name}'
        try:
            prob.solve(
                solver=cp.MOSEK,
                warm_start=True
                # eps=0.05,  # SCS convergence tolerance (1e-4)
                # max_iters=300,  # SCS max iterations (2500)
                # abstol=0.1, # ECOS (1e-8) / CVXOPT (1e-7) absolute accuracy
                # reltol=0.1 # ECOS (1e-8) / CVXOPT (1e-6) relative accuracy
            )
        except cp.error.SolverError as e:
            if log is not None:
                log.write(f'{name} encountered SolverError {str(e)}')
                log.write(f'{name} trying cp.SCS instead')
            prob.solve(solver=cp.SCS)

        if prob.status != 'optimal' and log is not None:
            log.write(f'{name} prob.status = {prob.status}')


def wrap_write_newlines(f: Any) -> Any:
    old_write = f.write

    def new_write(s):
        old_write(s + '\n')
        f.flush()
    f.write = new_write
    return f


def savefig(fig: plt.Figure, plots_dir: str, filename: str, **kwargs: Any) -> None:
    path = os.path.join(plots_dir, filename)
    defaults = dict(dpi=300, pad_inches=0, bbox_inches='tight', facecolor='white')
    fig.savefig(path, **(defaults | kwargs))


def load_pkl(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)
