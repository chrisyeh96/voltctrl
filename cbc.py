"""Convex body chasing code."""
from __future__ import annotations

from collections.abc import Callable

import cvxpy as cp
import numpy as np
from tqdm.auto import tqdm

from network_utils import is_pos_def, make_pd_and_pos

rng = np.random.default_rng()


def cp_triangle_norm_sq(x: cp.Expression) -> cp.Expression:
    return cp.norm(cp.upper_tri(x), 2)**2 + cp.norm(cp.diag(x), 2)**2


class CBCProjection:
    """Finds the set of X that is consistent with the observed data. Assumes
    that noise bound (eta) is known.

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
    TODO
    """
    def __init__(self, eta: float, n: int, T: int, n_samples: int, alpha: float,
                 v: np.ndarray,
                 gen_X_set: Callable[[cp.Variable], list[cp.Constraint]],
                 Vpar: tuple[np.ndarray, np.ndarray],
                 X_init: np.ndarray | None = None,
                 X_true: np.ndarray | None = None):
        """
        Args
        - eta: float, noise bound
        - n: int, # of buses
        - T: int, maximum # of time steps
        - n_samples: int, # of observations to use for defining the convex set
        - alpha: float, weight on slack variable
        - v: np.array, shape [n], initial squared voltage magnitudes
        - gen_X_set: function, takes an optimization variable (cp.Variable) and returns
            a list of constraints (cp.Constraint) describing the convex, compact
            uncertainty set for X
        - Vpar: tuple (Vpar_min, Vpar_max), box description of Vpar
            - each Vpar_* is a np.array of shape [n]
        - X_init: np.array, initial guess for X matrix, must be PSD and entry-wise >= 0
            - if None, we use X_init = np.eye(n)
        - X_true: np.array, true X matrix, optional
        """
        self.eta = eta
        self.n = n
        self.n_samples = n_samples
        self.alpha = alpha
        self.X_true = X_true

        # history
        self.vs = np.zeros([n, T+1])  # vs[:,t] = v(t)
        self.vs[:, 0] = v
        self.delta_vs = np.zeros([n, T])  # delta_vs[:,t] = v(t+1) - v(t)
        self.us = np.zeros([n, T])  # us[:,t] = u(t) = q^c(t+1) - q^c(t)
        self.qs = np.zeros([n, T+1])  # qs[:,t] = q^c(t)
        self.t = 0

        # define optimization variables
        self.var_X = cp.Variable([n, n], PSD=True)
        self.var_slack_w = cp.Variable(nonneg=True)  # nonneg=True

        self.X_set = gen_X_set(self.var_X)
        self.Vpar_min, self.Vpar_max = Vpar

        if X_init is None:
            X_init = np.eye(n)  # models a 1-layer tree graph  # TODO: check this!
        assert X_init.shape == (n, n)
        self.var_X.value = X_init  # this assignment will automatically check if X_init is PSD

        # project X_init into caligraphic X if necessary
        total_violation = sum(np.sum(constraint.violation()) for constraint in self.X_set)
        tqdm.write(f'X_init invalid. Violation: {total_violation:.3f}. Projecting into X_set.')
        obj = cp.Minimize(cp.norm(X_init - self.var_X))
        prob = cp.Problem(objective=obj, constraints=self.X_set)
        prob.solve(eps=0.05, max_iters=300)
        make_pd_and_pos(self.var_X.value)
        total_violation = sum(np.sum(constraint.violation()) for constraint in self.X_set)
        tqdm.write(f'After projection: X_init violation: {total_violation:.3f}.')

        self.X_init = self.var_X.value.copy()  # make a copy
        self.X_cache = self.var_X.value.copy()  # make a copy
        self.is_cached = True
        self.prob = None  # cp.Problem

    def add_obs(self, v: np.ndarray, u: np.ndarray) -> None:
        """
        Args
        - v: np.array, v(t+1)
        - u: np.array, u(t) = q^c(t+1) - q^c(t)
        """
        assert v.shape == (self.n,)
        assert u.shape == (self.n,)
        t = self.t
        self.vs[:, t+1] = v
        self.delta_vs[:, t] = v - self.vs[:, t]
        self.us[:, t] = u
        self.qs[:, t+1] = self.qs[:, t] + u
        self.t += 1
        self.is_cached = False

    def select(self) -> np.ndarray:
        """
        When select() is called, we have seen self.t observations. That is, we have values for:
          v(0), ...,   v(t)    # recall:   v(t) = vs[:, t]
        q^c(0), ..., q^c(t)    # recall: q^c(t) = qs[:, t]
          u(0), ...,   u(t-1)  # recall:   u(t) = us[:, t]
         Δv(0), ...,  Δv(t-1)  # recall:  Δv(t) = delta_vs[:, t]
        """
        if self.is_cached:
            return self.X_cache

        t = self.t
        assert t >= 1

        # be lazy if self.X_cache already satisfies the newest obs.
        w_hat = self.delta_vs[:, t-1] - self.X_cache @ self.us[:, t-1]
        vpar_hat = self.vs[:, t] - self.X_cache @ self.qs[:, t]
        w_hat_norm = np.max(np.abs(w_hat))
        if (w_hat_norm <= self.eta
                and np.all(self.Vpar_min <= vpar_hat)
                and np.all(vpar_hat <= self.Vpar_max)):
            # tqdm.write(f't = {self.t:6d}. CBC being lazy.')
            self.is_cached = True
            return self.X_cache
        tqdm.write(f't = {self.t:6d}, CBC pre opt ||ŵ(t)||∞: {w_hat_norm:.3f}')
        indent = ' ' * 11

        n = self.n
        ub = self.eta  # * np.ones([n, 1])
        lb = -ub

        vpar_min = self.Vpar_min.reshape(n, 1)
        vpar_max = self.Vpar_max.reshape(n, 1)

        # optimization variables
        X = self.var_X
        slack_w = self.var_slack_w

        # when t < self.n_samples, create a brand-new cp.Problem
        if t < self.n_samples:
            w_hats = self.delta_vs[:, 0:t] - X @ self.us[:, 0:t]
            vpar_hats = self.vs[:, 0:t+1] - X @ self.qs[:, 0:t+1]
            constrs = self.X_set + [
                lb - slack_w <= w_hats, w_hats <= ub + slack_w,
                vpar_min <= vpar_hats, vpar_hats <= vpar_max
            ]

            obj = cp.Minimize(cp_triangle_norm_sq(X - self.X_cache)
                              + self.alpha * slack_w)
            prob = cp.Problem(objective=obj, constraints=constrs)
            prob.solve(eps=0.1)

        # when t >= self.n_samples, compile a fixed-size optimization problem
        else:
            if self.prob is None:
                Xprev = cp.Parameter([n, n], PSD=True, name='Xprev')
                vs = cp.Parameter([n, self.n_samples], name='vs')
                delta_vs = cp.Parameter([n, self.n_samples], name='delta_vs')
                us = cp.Parameter([n, self.n_samples], name='us')
                qs = cp.Parameter([n, self.n_samples], name='qs')

                w_hats = delta_vs - X @ us
                vpar_hats = vs - X @ qs
                constrs = self.X_set + [
                    lb - slack_w <= w_hats, w_hats <= ub + slack_w,
                    vpar_min <= vpar_hats, vpar_hats <= vpar_max
                ]

                obj = cp.Minimize(cp_triangle_norm_sq(X-Xprev)
                                  + self.alpha * slack_w)
                self.prob = cp.Problem(objective=obj, constraints=constrs)

                # if cp.Problem is DPP, then it can be compiled for speedup
                # - http://cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming  # noqa
                tqdm.write(f'CBC prob is DPP?: {self.prob.is_dcp(dpp=True)}')

                self.param_Xprev = Xprev
                self.param_vs = vs
                self.param_delta_vs = delta_vs
                self.param_us = us
                self.param_qs = qs

            prob = self.prob

            # perform random sampling
            # - use the most recent k (<=5) time steps
            # - then sample additional previous time steps for self.n_samples total
            k = min(self.n_samples, 5)
            ts = np.concatenate([
                np.arange(t-k+1, t+1),
                rng.choice(t-k, size=self.n_samples-k, replace=False)])

            self.param_vs.value = self.vs[:, ts]
            self.param_delta_vs.value = self.delta_vs[:, ts]
            self.param_us.value = self.us[:, ts]
            self.param_qs.value = self.qs[:, ts]

            self.param_Xprev.value = self.X_cache
            prob.solve(warm_start=True,
                eps=0.05,  # SCS convergence tolerance (1e-4)
                max_iters=200,  # SCS max iterations (2500)
                # abstol=0.1, # ECOS (1e-8) / CVXOPT (1e-7) absolute accuracy
                # reltol=0.1 # ECOS (1e-8) / CVXOPT (1e-6) relative accuracy
            )

        if prob.status != 'optimal':
            tqdm.write(f'{indent} CBC prob.status = {prob.status}')
            if prob.status == 'infeasible':
                import pdb
                pdb.set_trace()
        self.X_cache = np.array(X.value)  # make a copy
        make_pd_and_pos(self.X_cache)
        self.is_cached = True

        # check slack variable
        if slack_w.value > 0:
            tqdm.write(f'{indent} CBC slack: {slack_w.value:.3f}')

        # check whether constraints are satisfied for latest time step
        w_hat = self.delta_vs[:, t-1] - self.X_cache @ self.us[:, t-1]
        vpar_hat = self.vs[:, t] - self.X_cache @ self.qs[:, t]
        w_hat_norm = np.max(np.abs(w_hat))
        tqdm.write(
            f'{indent} CBC post opt: '
            f'||ŵ(t)||∞: {w_hat_norm:.3f}, '
            f'max(0, vpar_min - vpar_hat): {max(0, np.max(self.Vpar_min - vpar_hat)):.3f}, '
            f'max(0, vpar_hat - vpar_max): {max(0, np.max(vpar_hat - self.Vpar_max)):.3f}')

        return np.array(self.X_cache)  # return a copy

        # TODO: calculate Steiner point?
        # if self.t > n + 1:
        #     steiner_point = psd_steiner_point(2, X, constraints)


class CBCProjectionWithNoise(CBCProjection):
    def __init__(self, eta: float, n: int, T: int, n_samples: int,
                 alpha: float, v: np.ndarray, X_init: np.ndarray | None = None,
                 X_true: np.ndarray | None = None):
        """
        Same args as CBCProjection. However, here, we interpret eta as an upper
        limit on true noise.
        """
        super().__init__(eta, n, T, n_samples, alpha, v, X_init, X_true)
        self.var_eta = cp.Variable(nonneg=True)
        self.eta_cache = 0

    def select(self) -> tuple[np.ndarray, float]:
        """
        When select() is called, we have seen self.t observations.
        """
        if self.is_cached:
            return self.X_cache, self.eta_cache

        t = self.t
        assert t >= 1

        # be lazy if self.X_cache already satisfies the newest obs.
        est_noise = self.delta_vs[:, t-1] - self.X_cache @ self.us[:, t-1]
        # tqdm.write(f'est_noise: {np.max(np.abs(est_noise)):.3f}')
        if np.max(np.abs(est_noise)) <= self.eta_cache:
            # buf = self.eta - np.max(np.abs(est_noise))
            # self.lazy_buffer.append(buf)
            # tqdm.write('being lazy')
            self.is_cached = True
            return self.X_cache, self.eta_cache

        n = self.n

        # optimization variables
        X = self.var_X
        slack = self.var_slack
        eta = self.var_eta

        ub = self.var_eta  # * np.ones([n, 1])
        lb = -ub

        # when t < self.n_samples, create a brand-new cp.Problem
        if t < self.n_samples:
            us = self.us[:, :t]
            delta_vs = self.delta_vs[:, :t]

            diffs = delta_vs - X @ us
            # constrs = [X >= 0, lb + slack <= diffs, diffs <= ub - slack]
            constrs = [X >= 0, lb <= diffs, diffs <= ub]

            obj = cp.Minimize(cp.norm(X - self.X_cache, 'fro'))
                # cp_triangle_norm_sq(X - self.X_cache)
                #               - self.alpha * slack)
            prob = cp.Problem(objective=obj, constraints=constrs)
            prob.solve(verbose=True)

        # when t >= self.n_samples, compile a fixed-size optimization problem
        else:
            if self.prob is None:
                Xprev = cp.Parameter([n, n], PSD=True, name='Xprev')
                us = cp.Parameter([n, self.n_samples], name='us')
                delta_vs = cp.Parameter([n, self.n_samples], name='delta_vs')

                diffs = delta_vs - X @ us
                constrs = [X >= 0, lb + slack <= diffs, diffs <= ub - slack]

                obj = cp.Minimize(cp_triangle_norm_sq(X-Xprev)
                                  - self.alpha * slack)
                self.prob = cp.Problem(objective=obj, constraints=constrs)

                # if CBC problem is DPP, then it can be compiled for speedup
                # - see https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming  # noqa
                tqdm.write(f'CBC prob is DPP?: {self.prob.is_dcp(dpp=True)}')

                self.param_Xprev = Xprev
                self.param_us = us
                self.param_delta_vs = delta_vs

            prob = self.prob

            # perform random sampling
            # - use the most recent k (<=5) time steps
            # - then sample additional previous time steps for 20 total
            k = min(self.n_samples, 5)
            ts = np.concatenate([
                np.arange(t-k, t),
                rng.choice(t-k, size=self.n_samples-k, replace=False)])
            self.param_us.value = self.us[:, ts]
            self.param_delta_vs.value = self.delta_vs[:, ts]

            self.param_Xprev.value = self.X_cache
            prob.solve(warm_start=True)

        if prob.status != 'optimal':
            tqdm.write(f'CBC prob.status = {prob.status}')
            if prob.status == 'infeasible':
                import pdb
                pdb.set_trace()
        self.X_cache = np.array(X.value)  # make a copy

        if np.any(self.X_cache < 0):
            tqdm.write(f'optimal X has neg values. min={np.min(self.X_cache)}')
            tqdm.write('- applying ReLu')
            self.X_cache = np.maximum(0, self.X_cache)

        self.is_cached = True
        return self.X_cache

        # TODO: calculate Steiner point?
        # if self.t > n + 1:
        #     steiner_point = psd_steiner_point(2, X, constraints)


def psd_steiner_point(num_samples, X, constraints) -> np.ndarray:
    """
    Args
    - num_samples: int, number of samples to use for calculating Steiner point
    - X: cp.Variable, shape [n, n]
    - constraints: list, cvxpy constraints
    """
    n = X.shape[0]

    S = 0
    for i in range(num_samples):
        theta = rng.random(X.shape)
        theta = theta @ theta.T + 1e-7 * np.eye(n)  # random strictly PD matrix
        theta /= np.linalg.norm(theta, 'fro')  # unit norm

        objective = cp.Maximize(cp.trace(theta @ X))
        prob = cp.Problem(objective=objective, constraints=constraints)
        prob.solve()
        assert prob.status == 'optimal'

        p_i = prob.value
        S += p_i * theta

    d = n + n*(n-1) // 2
    S = S / num_samples * d

    # check to make sure there is no constraint violation
    X.value = S
    for constr in constraints:
        constr.violation()
