"""Convex body chasing code."""
from __future__ import annotations

import cvxpy as cp
import numpy as np
from tqdm.auto import tqdm

rng = np.random.default_rng()


def cp_triangle_norm_sq(x: cp.Expression) -> cp.Expression:
    return cp.norm(cp.upper_tri(x), 2)**2 + cp.norm(cp.diag(x), 2)**2


class CBCProjection:
    """Finds the set of X that is consistent with the observed data.
    """
    def __init__(self, eta: float, n: int, T: int, n_samples: int,
                 alpha: float, v: np.ndarray, X_init: np.ndarray | None = None,
                 X_true: np.ndarray | None = None):
        """
        Args
        - eta: float, noise bound
        - n: int, # of buses
        - T: int, maximum # of time steps
        - n_samples: int, # of observations to use for defining the convex set
        - alpha: float, weight on slack variable
        - v: np.array, shape [n], initial squared voltage magnitudes
        - X_init: np.array, initial guess for X matrix, must be PSD and
            entry-wise >= 0
            - if None, we use X_init = np.eye(n)
        - X_true: np.array, true X matrix, optional
        """
        self.eta = eta
        self.n = n
        self.n_samples = n_samples
        self.alpha = alpha
        self.X_true = X_true

        # history
        self.delta_vs = np.zeros([n, T+1])
        self.v_prev = v
        self.us = np.zeros([n, T])
        self.t = 0

        if X_init is None:
            X_init = np.eye(n)  # models a 1-layer tree graph
        self.X_cache = X_init
        self.is_cached = True
        self.lazy_buffer = []
        self.prob = None

        # define optimization variables
        self.var_X = cp.Variable([n, n], PSD=True)
        self.var_slack = cp.Variable(nonneg=True)

    def add_obs(self, v: np.ndarray, u: np.ndarray) -> None:
        assert v.shape == (self.n,)
        assert u.shape == (self.n,)
        self.us[:, self.t] = u
        self.delta_vs[:, self.t] = v - self.v_prev
        self.t += 1
        self.v_prev = v
        self.is_cached = False

    def select(self) -> np.ndarray:
        """
        When select() is called, we have seen self.t observations.
        """
        if self.is_cached:
            return self.X_cache

        t = self.t
        assert t >= 1

        # be lazy if self.X_cache already satisfies the newest obs.
        est_noise = self.delta_vs[:, t-1] - self.X_cache @ self.us[:, t-1]
        tqdm.write(f'est_noise: {np.max(np.abs(est_noise)):.3f}')
        if np.max(np.abs(est_noise)) <= self.eta:
            # buf = self.eta - np.max(np.abs(est_noise))
            # self.lazy_buffer.append(buf)
            # tqdm.write('being lazy')
            self.is_cached = True
            return self.X_cache

        tqdm.write('not lazy')

        n = self.n
        ub = self.eta  # * np.ones([n, 1])
        lb = -ub

        # optimization variables
        X = self.var_X
        slack = self.var_slack

        # import pdb
        # pdb.set_trace()

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
