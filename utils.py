"""Convex body chasing base class + utilities."""
from __future__ import annotations

import io

import cvxpy as cp
from tqdm.auto import tqdm
from typing import Any


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
