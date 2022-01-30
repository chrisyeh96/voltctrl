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
        self.v = np.zeros([T+1, n])  # v[t] = v(t)
        self.v[0] = v
        self.delta_v = np.zeros([T, n])  # delta_v[t] = v(t+1) - v(t)
        self.u = np.zeros([T, n])  # u[t] = u(t) = q^c(t+1) - q^c(t)
        self.q = np.zeros([T+1, n])  # q[t] = q^c(t)
        self.t = 0

        self.w_inds = np.zeros([2, T], dtype=bool)
        self.vpar_inds = np.zeros([2, T+1], dtype=bool)
        self.w_inds[:, 0] = True
        self.vpar_inds[:, 1] = True

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
        # prob.solve(eps=0.05, max_iters=300)
        prob.solve(solver=cp.MOSEK)
        make_pd_and_pos(self.var_X.value)
        total_violation = sum(np.sum(constraint.violation()) for constraint in self.X_set)
        tqdm.write(f'After projection: X_init violation: {total_violation:.3f}.')

        self.X_init = self.var_X.value.copy()  # make a copy
        self.X_cache = self.var_X.value.copy()  # make a copy
        self.is_cached = True
        self._setup_prob()

    def _setup_prob(self):
        n = self.n
        ub = self.eta  # * np.ones([n, 1])
        lb = -ub

        # optimization variables
        X = self.var_X
        slack_w = self.var_slack_w

        constrs = self.X_set
        self.param = {}

        Xprev = cp.Parameter([n, n], PSD=True, name='Xprev')
        for b in ['lb', 'ub']:
            vs = cp.Parameter([self.n_samples, n], name=f'vs_{b}')
            delta_vs = cp.Parameter([self.n_samples, n], name=f'delta_vs_{b}')
            us = cp.Parameter([self.n_samples, n], name=f'us_{b}')
            qs = cp.Parameter([self.n_samples, n], name=f'qs_{b}')

            w_hats = delta_vs - us @ X
            vpar_hats = vs - qs @ X

            if b == 'lb':
                constrs.extend([lb - slack_w <= w_hats,
                                self.Vpar_min[None, :] <= vpar_hats])
            else:
                constrs.extend([w_hats <= ub + slack_w,
                                vpar_hats <= self.Vpar_max[None, :]])

            self.param[f'vs_{b}'] = vs
            self.param[f'delta_vs_{b}'] = delta_vs
            self.param[f'us_{b}'] = us
            self.param[f'qs_{b}'] = qs
        self.param['Xprev'] = Xprev

        # constrs = self.X_set + [
        #     lb - slack_w <= w_hats, w_hats <= ub + slack_w,
        #     self.Vpar_min[None, :] <= vpar_hats, vpar_hats <= self.Vpar_max[None, :]
        # ]

        obj = cp.Minimize(cp_triangle_norm_sq(X-Xprev)
                          + self.alpha * slack_w)
        self.prob = cp.Problem(objective=obj, constraints=constrs)

        # if cp.Problem is DPP, then it can be compiled for speedup
        # - http://cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming  # noqa
        tqdm.write(f'CBC prob is DPP?: {self.prob.is_dcp(dpp=True)}')

        # self.param_Xprev = Xprev
        # self.param_vs = vs
        # self.param_delta_vs = delta_vs
        # self.param_us = us
        # self.param_qs = qs

    def _check_informative(self, t: int, b: np.ndarray, c: np.ndarray,
                           useful: np.ndarray) -> None:
        """
        Args
        - t: int
        - b, c: np.ndarray, shape [T, n]
        - useful: np.ndarray, shape [2, T], boolean indexing vector
            - 1st row is for lower bound, 2nd row is for upper bound
        """
        # manage contstraints of the form: d <= b - X c
        # - each previous point (b',c') is useful if (b' ⋡ b) or (c' ⋠ c)
        # - new point is useful if no other point has (b' ≼ b and c' ≽ c)
        useful_lb = useful[0]
        cmp_b = (b[t] >= b[useful_lb])
        cmp_c = (c[t] <= c[useful_lb])
        useful_lb[useful_lb] = np.any(cmp_b, axis=1) | np.any(cmp_c, axis=1)
        useful_lb[t] = ~np.any(np.all(cmp_b, axis=1) & np.all(cmp_c, axis=1))

        # manage constraints of the form: b - X c <= d
        # - each previous point (b',c') is useful if (b' ⋠ b) or (c' ⋡ c)
        # - new point is useful if no other point has (b' ≽ b and c' ≼ c)
        useful_ub = useful[1]
        cmp_b = (b[t] <= b[useful_ub])
        cmp_c = (c[t] >= c[useful_ub])
        useful_ub[useful_ub] = np.any(cmp_b, axis=1) | np.any(cmp_c, axis=1)
        useful_ub[t] = ~np.any(np.all(cmp_b, axis=1) & np.all(cmp_c, axis=1))


    def add_obs(self, v: np.ndarray, u: np.ndarray) -> None:
        """
        Args
        - v: np.array, v(t+1) = v(t) + X @ u(t) = X @ q^c(t+1) + vpar(t+1)
        - u: np.array, u(t) = q^c(t+1) - q^c(t)
        """
        assert v.shape == (self.n,)
        assert u.shape == (self.n,)
        t = self.t
        q = self.q[t] + u
        delta_v = v - self.v[t]
        self.u[t] = u
        self.delta_v[t] = delta_v
        self.v[t+1] = v
        self.q[t+1] = q

        if t >= 1:
            self._check_informative(t=t, b=self.delta_v, c=self.u, useful=self.w_inds)
            self._check_informative(t=t+1, b=self.v, c=self.q, useful=self.vpar_inds)

            # cmp_delta = (delta_v <= self.delta_v[self.w_inds_ub])
            # cmp_u = (u >= self.us[self.w_inds_ub])
            # self.w_inds_ub[self.w_inds_ub] = np.any(cmp_delta, axis=1) | np.any(cmp_u, axis=1)
            # self.w_inds_ub[t] = ~np.any(np.all(cmp_delta, axis=1) & np.all(cmp_u, axis=1))

            # cmp_v = (v <= self.v[self.vpar_inds_ub])
            # cmp_q = (q >= self.q[self.vpar_inds_ub])
            # self.vpar_inds_ub[self.self.vpar_inds_ub] = np.any(cmp_v, axis=1) | np.any(cmp_u, axis=1)
            # self.vpar_inds_ub[t] = ~np.any(np.all(cmp_delta, axis=1) & np.all(cmp_u, axis=1))

            if (t+1) % 500 == 0:
                num_w_inds = tuple(np.sum(self.w_inds[:t+1], axis=1))
                num_vpar_inds = tuple(np.sum(self.vpar_inds[:t+2], axis=1))
                tqdm.write(f'active constraints - w: {num_w_inds}/{t+1}, vpar: {num_vpar_inds}/{t+1}')

        self.t += 1
        self.is_cached = False

    def _check_newest_obs(self) -> tuple[bool, str]:
        """Checks whether self.X_cache satisfies the newest observation.

        Returns:
        - satisfied: bool, whether self.X_cache satisfies the newest observation
        - msg: str, (if not satisfied) describes which constraints are not satisfied,
            (if satisfied) is empty string ''
        """
        t = self.t

        w_hat = self.delta_v[t-1] - self.u[t-1] @ self.X_cache
        vpar_hat = self.v[t] - self.q[t] @ self.X_cache
        w_hat_norm = np.max(np.abs(w_hat))

        vpar_lower_violation = np.max(self.Vpar_min - vpar_hat)
        vpar_upper_violation = np.max(vpar_hat - self.Vpar_max)

        msgs = []
        if w_hat_norm > self.eta:
            msgs.append(f'||ŵ(t)||∞: {w_hat_norm:.3f}')
        if vpar_lower_violation > 0.05:
            msgs.append(f'max(vpar_min - vpar_hat): {vpar_lower_violation:.3f}')
        if vpar_upper_violation > 0.05:
            msgs.append(f'max(vpar_hat - vpar_max): {vpar_upper_violation:.3f}')
        satisfied = (len(msgs) == 0)
        msg = ', '.join(msgs)
        return satisfied, msg

    def select(self) -> np.ndarray:
        """
        When select() is called, we have seen self.t observations. That is, we have values for:
          v(0), ...,   v(t)    # recall:   v(t) = vs[t]
        q^c(0), ..., q^c(t)    # recall: q^c(t) = qs[t]
          u(0), ...,   u(t-1)  # recall:   u(t) = us[t]
         Δv(0), ...,  Δv(t-1)  # recall:  Δv(t) = delta_vs[t]
        """
        if self.is_cached:
            return self.X_cache

        t = self.t
        assert t >= 1

        # be lazy if self.X_cache already satisfies the newest obs.
        satisfied, msg = self._check_newest_obs()
        if satisfied:
            # tqdm.write(f't = {self.t:6d}. CBC being lazy.')
            self.is_cached = True
            return self.X_cache
        tqdm.write(f't = {self.t:6d}, CBC pre opt: {msg}')
        indent = ' ' * 11

        n = self.n
        ub = self.eta  # * np.ones([n, 1])
        lb = -ub

        # optimization variables
        X = self.var_X
        slack_w = self.var_slack_w

        # when t < self.n_samples
        if t < self.n_samples:
            for b in ['lb', 'ub']:
                self.param[f'vs_{b}'].value = np.tile(self.Vpar_min, [self.n_samples, 1])
                self.param[f'vs_{b}'].value[:t] = self.v[1:1+t]
                self.param[f'delta_vs_{b}'].value = self.delta_v[:self.n_samples]
                self.param[f'us_{b}'].value = self.u[:self.n_samples]
                self.param[f'qs_{b}'].value = self.q[1:1+self.n_samples]

        # when t >= self.n_samples
        else:
            # perform random sampling
            # - use the most recent k time steps  [t-k, ..., t-1]
            # - then sample additional previous time steps for self.n_samples total
            #   [0, ..., t-k-1]
            k = min(self.n_samples, 20)
            # ts = np.concatenate([
            #     np.arange(t-k, t),
            #     rng.choice(t-k, size=self.n_samples-k, replace=False)])

            for i, b in enumerate(['lb', 'ub']):
                w_inds = self.w_inds[i].nonzero()[0]
                ts = np.concatenate([
                    w_inds[-k:],
                    rng.choice(len(w_inds) - k, size=self.n_samples-k, replace=False)
                ])
                self.param[f'delta_vs_{b}'].value = self.delta_v[ts]
                self.param[f'us_{b}'].value = self.u[ts]

                vpar_inds = self.vpar_inds[i].nonzero()[0]
                ts = np.concatenate([
                    vpar_inds[-k:],
                    rng.choice(len(vpar_inds) - k, size=self.n_samples-k, replace=False)
                ])
                self.param[f'vs_{b}'].value = self.v[ts]
                self.param[f'qs_{b}'].value = self.q[ts]

            # self.param_vs.value = self.v[ts+1]
            # self.param_delta_vs.value = self.delta_v[ts]
            # self.param_us.value = self.us[ts]
            # self.param_qs.value = self.q[ts+1]

        # self.param_Xprev.value = self.X_cache
        self.param['Xprev'].value = self.X_cache

        prob = self.prob
        prob.solve(
            solver=cp.MOSEK,
            warm_start=True,
            # eps=0.05,  # SCS convergence tolerance (1e-4)
            # max_iters=300,  # SCS max iterations (2500)
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
        satisfied, msg = self._check_newest_obs()
        if not satisfied:
            tqdm.write(f'{indent} CBC post opt: {msg}')

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
        est_noise = self.delta_v[:, t-1] - self.X_cache @ self.u[:, t-1]
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
            delta_vs = self.delta_v[:, :t]

            diffs = delta_vs - X @ us
            constrs = [X >= 0, lb <= diffs, diffs <= ub, eta <= self.eta]
            # constrs = [X >= 0, lb + slack <= diffs, diffs <= ub - slack,
            #            eta <= self.eta]

            obj = cp.Minimize(cp_triangle_norm_sq(X - self.X_cache)
                              + 1e3 * eta**2)
            # obj = cp.Minimize(cp_triangle_norm_sq(X - self.X_cache)
            #                   + (eta - self.eta_cache)**2 + eta**2)
            # obj = cp.Minimize(cp_triangle_norm_sq(X - self.X_cache)
            #                   + (eta - self.eta_cache)**2
            #                   - self.alpha * slack)
            prob = cp.Problem(objective=obj, constraints=constrs)
            prob.solve()

        # when t >= self.n_samples, compile a fixed-size optimization problem
        else:
            if self.prob is None:
                Xprev = cp.Parameter([n, n], nonneg=True, name='Xprev')
                etaprev = cp.Parameter(nonneg=True, name='eta')
                us = cp.Parameter([n, self.n_samples], name='us')
                delta_vs = cp.Parameter([n, self.n_samples], name='delta_vs')

                diffs = delta_vs - X @ us
                # constrs = [X >= 0, lb <= diffs, diffs <= ub, eta <= self.eta]
                constrs = [X >= 0, lb + slack <= diffs, diffs <= ub - slack,
                           etaprev <= eta, eta <= self.eta]

                obj = cp.Minimize(cp_triangle_norm_sq(X - Xprev)
                                  + 3e2 * eta**2
                                  - self.alpha * slack)
                # obj = cp.Minimize(cp_triangle_norm_sq(X - Xprev)
                #                   + (eta - etaprev)**2)
                # obj = cp.Minimize(cp_triangle_norm_sq(X - Xprev)
                #                   + (eta - etaprev)**2
                #                   - self.alpha * slack)
                self.prob = cp.Problem(objective=obj, constraints=constrs)

                # if CBC problem is DPP, then it can be compiled for speedup
                # - see https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming  # noqa
                tqdm.write(f'CBC prob is DPP?: {self.prob.is_dcp(dpp=True)}')

                self.param_Xprev = Xprev
                self.param_etaprev = etaprev
                self.param_us = us
                self.param_delta_vs = delta_vs

            prob = self.prob

            # perform random sampling
            # - use the most recent k (<=5) time steps
            # - then sample additional previous time steps for 20 total
            k = min(self.n_samples, 5)
            sample_probs = np.linalg.norm(
                self.delta_v[:, :t-k] - self.X_cache @ self.us[:, :t-k],
                axis=0)
            sample_probs /= np.sum(sample_probs)
            ts = np.concatenate([
                np.arange(t-k, t),
                rng.choice(t-k, size=self.n_samples-k, replace=False,
                           p=sample_probs)
            ])
            self.param_us.value = self.us[:, ts]
            self.param_delta_vs.value = self.delta_v[:, ts]

            self.param_Xprev.value = self.X_cache
            self.param_etaprev.value = self.eta_cache
            prob.solve(warm_start=True)

        if prob.status != 'optimal':
            tqdm.write(f'CBC prob.status = {prob.status}')
            if prob.status == 'infeasible':
                import pdb
                pdb.set_trace()
        self.X_cache = np.array(X.value)  # make a copy
        self.eta_cache = float(eta.value)  # make a copy

        # Force symmetry, even if all-close. But only print error message if
        # not all-close.
        if not np.allclose(self.X_cache, self.X_cache.T):
            max_diff = np.max(np.abs(self.X_cache - self.X_cache.T))
            tqdm.write(f'optimal X not symmetric. ||X-X.T||_max = {max_diff}'
                       ' - making symmetric')
        self.X_cache = (self.X_cache + self.X_cache.T) / 2

        # check for PSD
        w, V = np.linalg.eigh(self.X_cache)
        if np.any(w < 0):
            tqdm.write(f'optimal X not PSD. smallest eigenvalue = {np.min(w)}'
                       ' - setting neg eigenvalues to 0')
            w[w < 0] = 0
            self.X_cache = (V * w) @ V.T

        if np.any(self.X_cache < 0):
            tqdm.write(f'optimal X has neg values. min={np.min(self.X_cache)}'
                       ' - applying ReLu')
            self.X_cache = np.maximum(0, self.X_cache)

        self.is_cached = True
        return (self.X_cache, self.eta_cache)

        # TODO: calculate Steiner point?
        # if self.t > n + 1:
        #     steiner_point = psd_steiner_point(2, X, constraints)


def psd_steiner_point(num_samples, X, constraints) -> np.ndarray:
    """
    Args
    - num_samples: int, number of samples to use for calculating Steiner point
    - X: cp.Variable, shape [n, n]
    - constraints: list, cvxpy constraints on X
    """
    n = X.shape[0]
    S = 0

    param_theta = cp.Parameter(X.shape)
    objective = cp.Maximize(cp.trace(param_theta @ X))
    prob = cp.Problem(objective=objective, constraints=constraints)
    assert prob.is_dcp(dpp=True)

    for i in range(num_samples):
        theta = rng.random(X.shape)
        theta = theta @ theta.T + 1e-7 * np.eye(n)  # random strictly PD matrix
        theta /= np.linalg.norm(theta, 'fro')  # unit norm

        param_theta.value = theta
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
