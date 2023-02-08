"""Convex body chasing base class + utilities."""
from __future__ import annotations

from collections.abc import Callable, Sequence
import io

import cvxpy as cp
import numpy as np
from tqdm.auto import tqdm

from network_utils import make_pd_and_pos, np_triangle_norm

Constraint = cp.constraints.constraint.Constraint


def cp_triangle_norm_sq(x: cp.Expression) -> cp.Expression:
    return cp.norm(cp.upper_tri(x), 2)**2 + cp.norm(cp.diag(x), 2)**2


def project_into_X_set(X_init: np.ndarray, var_X: cp.Variable,
                       log: tqdm | io.TextIOBase | None,
                       X_set: list[Constraint], X_true: np.ndarray) -> None:
    """Project X_init into 𝒳 if necessary."""
    if log is not None:
        norm = np_triangle_norm(X_init)
        dist = np_triangle_norm(X_init - X_true)
        log.write(f'X_init: ||X̂||_△ = {norm:.3f}, ||X̂-X||_△ = {dist:.3f}')

    var_X.value = X_init  # if var_X.is_psd(), this automatically checks that X_init is PSD
    total_violation = sum(np.sum(constraint.violation()) for constraint in X_set)
    if total_violation == 0 and log is not None:
        log.write('X_init valid.')
    else:
        if log is not None:
            log.write(f'X_init invalid. Violation: {total_violation:.3f}. Projecting into 𝒳.')
        obj = cp.Minimize(cp_triangle_norm_sq(X_init - var_X))
        prob = cp.Problem(objective=obj, constraints=X_set)
        try:
            prob.solve(solver=cp.MOSEK)
        except cp.error.SolverError as e:
            if log is not None:
                log.write(str(e))
            prob.solve(solver=cp.SCS)
        make_pd_and_pos(var_X.value)
        if log is not None:
            total_violation = sum(np.sum(constraint.violation()) for constraint in X_set)
            norm = np_triangle_norm(var_X.value)
            dist = np_triangle_norm(var_X.value - X_true)
            log.write(f'After projection: X_init violation: {total_violation:.3f}.')
            log.write(f'                  ||X̂||_△ = {norm:.3f}, ||X̂-X||_△ = {dist:.3f}')


class CBCBase:
    """Base class for Consistent Model Chasing.

    In our implementation, we use a different indexing system than the paper.
    For, t = 0, ..., T:
    v(t) = X q^c(t) + vpar(t)
    vpar(t) = X q^e(t) + R p(t) + v^0
    u(t) = q^c(t-1) - q^c(t)
    v(t+1) = v(t) + X u(t) + w(t)
    w(t) = v(t+1) - v(t) - X u(t)
         = vpar(t+1) - vpar(t)
         = X[q^e(t+1) - q^e(t)] + R[p(t+1) - p(t)]

    We assume that q^c(0) = 0, and that v(0) is given.

    Usage:
        sel = CBCBase(n, T, X_init, v, ...)  # initialize
        for t in range(T-1):
            Xhat = sel.select(t)
            q_next = get_control_action(Xhat, ...)

            sel.q[t+1] = q_next
            sel.v[t+1] = X_true @ q_next + vpar_true[t+1]
            sel.update(t+1)
    """
    def __init__(self, n: int, T: int, X_init: np.ndarray, v: np.ndarray,
                 gen_X_set: Callable[[cp.Variable], list[Constraint]],
                 X_true: np.ndarray,
                 obs_nodes: Sequence[int] | None = None,
                 log: tqdm | io.TextIOBase | None = None):
        """
        Args
        - n: int, # of buses
        - T: int, maximum # of time steps
        - X_init: np.array, shape [n, n], initial guess for X matrix, must be
            PSD and entry-wise >= 0
        - v: np.array, shape [n], initial squared voltage magnitudes
        - gen_X_set: function, takes an optimization variable (cp.Variable) and returns
            a list of constraints (cp.Constraint) describing the convex, compact
            uncertainty set for X
        - X_true: np.array, shape [n, n], true X matrix, optional
        - obs_nodes: list of int, nodes that we can observe voltages for
        - log: object with .write() function, defaults to tqdm
        """
        self.n = n
        self.X_init = X_init
        self.X_true = X_true

        if log is None:
            log = tqdm
        self.log = log

        # history
        self.v = np.zeros([T, n])  # v[t] = v(t)
        self.v[0] = v
        self.delta_v = np.zeros([T-1, n])  # delta_v[t] = v(t+1) - v(t)
        self.u = np.zeros([T-1, n])  # u[t] = u(t) = q^c(t+1) - q^c(t)
        self.q = np.zeros([T, n])  # q[t] = q^c(t)

        # define optimization variables
        self.var_X = cp.Variable((n, n), PSD=True)
        assert X_init.shape == (n, n)

        self.X_set = gen_X_set(self.var_X)
        self._init_X(X_init)
        self.X_init = self.var_X.value.copy()  # make a copy
        self.X_cache = self.var_X.value.copy()  # make a copy

        # handle observable nodes
        if obs_nodes is None:
            obs_nodes = list(range(n))
        self.obs_nodes = obs_nodes

    def _init_X(self, X_init: np.ndarray) -> None:
        project_into_X_set(X_init=X_init, var_X=self.var_X,
                           log=self.log, X_set=self.X_set,
                           X_true=self.X_true)

    def add_obs(self, t: int) -> None:
        """Process new observation.

        Args
        - t: int, current time step, v[t] and q[t] have just been updated
        """
        assert t >= 1
        self.u[t-1] = self.q[t] - self.q[t-1]
        self.delta_v[t-1] = self.v[t] - self.v[t-1]

    def select(self, t: int) -> np.ndarray:
        """
        Args
        - t: int, current time step

        When select() is called, we have seen t observations. That is, we have values for:
          v(0), ...,   v(t)    # recall:   v(t) = vs[t]
        q^c(0), ..., q^c(t)    # recall: q^c(t) = qs[t]
          u(0), ...,   u(t-1)  # recall:   u(t) = us[t]
         Δv(0), ...,  Δv(t-1)  # recall:  Δv(t) = delta_vs[t]
        """
        return self.X_init
