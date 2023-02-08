"""Convex body chasing via projection."""
from __future__ import annotations

from collections.abc import Callable, Sequence
import io

import cvxpy as cp
import numpy as np
from tqdm.auto import tqdm

from cbc.base import CBCBase, cp_triangle_norm_sq
from network_utils import make_pd_and_pos

Constraint = cp.constraints.constraint.Constraint


class CBCProjection(CBCBase):
    """Finds the set of X that is consistent with the observed data. Assumes
    that noise bound (eta) is known.
    """
    def __init__(self, n: int, T: int, X_init: np.ndarray, v: np.ndarray,
                 gen_X_set: Callable[[cp.Variable], list[Constraint]],
                 eta: float, nsamples: int, alpha: float,
                 Vpar: tuple[np.ndarray, np.ndarray],
                 X_true: np.ndarray,
                 obs_nodes: Sequence[int] | None = None,
                 log: tqdm | io.TextIOBase | None = None, seed: int = 123):
        """
        Args
        - see CBCBase for descriptions of other parameters
        - eta: float, noise bound
        - nsamples: int, # of observations to use for defining the convex set
        - alpha: float, weight on slack variable
        - Vpar: tuple (Vpar_min, Vpar_max), box description of Vpar
            - each Vpar_* is a np.array of shape [n]
        - seed: int, random seed
        """
        super().__init__(n=n, T=T, X_init=X_init, v=v, gen_X_set=gen_X_set,
                         X_true=X_true, obs_nodes=obs_nodes, log=log)
        self.is_cached = True

        self.eta = eta
        self.nsamples = nsamples
        self.alpha = alpha

        self.w_inds = np.zeros([2, T-1], dtype=bool)  # whether each (u(t), delta_v(t)) is useful
        self.vpar_inds = np.zeros([2, T], dtype=bool)  # whether each (v(t), q(t)) is useful
        self.w_inds[:, 0] = True
        self.vpar_inds[:, 1] = True

        self.var_slack_w = cp.Variable(nonneg=True)  # nonneg=True
        self.Vpar_min, self.Vpar_max = Vpar

        self._setup_prob()
        self.rng = np.random.default_rng(seed)

    def _setup_prob(self) -> None:
        """Defines self.prob as the projection of Xprev into the consistent set.
        """
        n = self.n
        ub = self.eta  # * np.ones([n, 1])
        lb = -ub

        # optimization variables
        X = self.var_X
        slack_w = self.var_slack_w

        constrs = self.X_set
        self.param = {}

        Xprev = cp.Parameter((n, n), PSD=True, name='Xprev')
        for b in ['lb', 'ub']:
            vs = cp.Parameter((self.nsamples, n), name=f'vs_{b}')
            delta_vs = cp.Parameter((self.nsamples, n), name=f'delta_vs_{b}')
            us = cp.Parameter((self.nsamples, n), name=f'us_{b}')
            qs = cp.Parameter((self.nsamples, n), name=f'qs_{b}')

            w_hats = delta_vs - us @ X
            vpar_hats = vs - qs @ X

            if b == 'lb':
                constrs.extend([
                    lb - slack_w <= w_hats,
                    self.Vpar_min[None, self.obs_nodes] <= vpar_hats[:, self.obs_nodes]
                ])
            else:
                constrs.extend([
                    w_hats <= ub + slack_w,
                    vpar_hats[:, self.obs_nodes] <= self.Vpar_max[None, self.obs_nodes]
                ])

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
        self.log.write(f'CBC prob is DPP?: {self.prob.is_dcp(dpp=True)}')

        # self.param_Xprev = Xprev
        # self.param_vs = vs
        # self.param_delta_vs = delta_vs
        # self.param_us = us
        # self.param_qs = qs

    def _check_informative(self, t: int, b: np.ndarray, c: np.ndarray,
                           useful: np.ndarray) -> None:
        """Checks whether b[t], c[t] are useful.

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

    def add_obs(self, t: int) -> None:
        """
        Args
        - t: int, current time step (>=1), v[t] and q[t] have just been updated

        Args
        - v: np.array, v(t+1) = v(t) + X @ u(t) = X @ q^c(t+1) + vpar(t+1)
        - u: np.array, u(t) = q^c(t+1) - q^c(t)
        """
        # update self.u and self.delta_v
        super().add_obs(t)

        if self.is_cached:
            satisfied, msg = self._check_newest_obs(t)
            if not satisfied:
                self.is_cached = False
                self.log.write(f't = {t:6d}, CBC pre opt: {msg}')

        if t >= 2:
            self._check_informative(t=t-1, b=self.delta_v, c=self.u, useful=self.w_inds)
            self._check_informative(t=t, b=self.v, c=self.q, useful=self.vpar_inds)

        # cmp_delta = (delta_v <= self.delta_v[self.w_inds_ub])
        # cmp_u = (u >= self.us[self.w_inds_ub])
        # self.w_inds_ub[self.w_inds_ub] = np.any(cmp_delta, axis=1) | np.any(cmp_u, axis=1)
        # self.w_inds_ub[t] = ~np.any(np.all(cmp_delta, axis=1) & np.all(cmp_u, axis=1))

        # cmp_v = (v <= self.v[self.vpar_inds_ub])
        # cmp_q = (q >= self.q[self.vpar_inds_ub])
        # self.vpar_inds_ub[self.self.vpar_inds_ub] = np.any(cmp_v, axis=1) | np.any(cmp_u, axis=1)
        # self.vpar_inds_ub[t] = ~np.any(np.all(cmp_delta, axis=1) & np.all(cmp_u, axis=1))

        if t % 500 == 0:
            num_w_inds = tuple(np.sum(self.w_inds[:t], axis=1))
            num_vpar_inds = tuple(np.sum(self.vpar_inds[:t+1], axis=1))
            self.log.write(f'active constraints - w: {num_w_inds}/{t}, vpar: {num_vpar_inds}/{t}')

    def _check_newest_obs(self, t: int) -> tuple[bool, str]:
        """Checks whether self.X_cache satisfies the newest observation:
        (v[t], q[t], u[t-1], delta_v[t-1])

        Returns
        - satisfied: bool, whether self.X_cache satisfies the newest observation
        - msg: str, (if not satisfied) describes which constraints are not satisfied,
            (if satisfied) is empty string ''
        """
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

    def select(self, t: int) -> np.ndarray:
        """
        We have seen t observations. That is, we have values for:
          v(0), ...,   v(t)    # recall:   v(t) = vs[t]
        q^c(0), ..., q^c(t)    # recall: q^c(t) = qs[t]
          u(0), ...,   u(t-1)  # recall:   u(t) = us[t]
         Δv(0), ...,  Δv(t-1)  # recall:  Δv(t) = delta_vs[t]

        It is possible that t=0, meaning we haven't seen any observations yet.
        (We have v(0) and q^c(0), but not u(0) or Δv(0).) In this case, our
        X_init should be cached, and we will return that.

        Args
        - t: int, current time step (>=0)
        """
        # be lazy if self.X_cache already satisfies the newest obs.
        if self.is_cached:
            return self.X_cache

        indent = ' ' * 11

        n = self.n
        ub = self.eta  # * np.ones([n, 1])
        lb = -ub

        # optimization variables
        # - assuming that assumptions 1, 2, and the first part of 3
        #     ($\forall t: \vpar(t) \in \Vpar$) are satisfied, we don't need
        #     actually need a slack variable in SEL (the CBC algorithm)
        X = self.var_X
        slack_w = self.var_slack_w

        # when t < self.nsamples
        if t < self.nsamples:
            for b in ['lb', 'ub']:
                self.param[f'vs_{b}'].value = np.tile(self.Vpar_min, [self.nsamples, 1])
                self.param[f'vs_{b}'].value[:t] = self.v[1:1+t]
                self.param[f'delta_vs_{b}'].value = self.delta_v[:self.nsamples]
                self.param[f'us_{b}'].value = self.u[:self.nsamples]
                self.param[f'qs_{b}'].value = self.q[1:1+self.nsamples]

        # when t >= self.nsamples
        else:
            # perform random sampling
            # - use the most recent k time steps  [t-k, ..., t-1]
            # - then sample additional previous time steps for self.nsamples total
            #   [0, ..., t-k-1]
            k = min(self.nsamples, 20)
            # ts = np.concatenate([
            #     np.arange(t-k, t),
            #     rng.choice(t-k, size=self.nsamples-k, replace=False)])

            for i, b in enumerate(['lb', 'ub']):
                w_inds = self.w_inds[i, :t].nonzero()[0]
                ts = np.concatenate([
                    w_inds[-k:],
                    self.rng.choice(len(w_inds) - k, size=self.nsamples-k, replace=False)
                ])
                self.param[f'delta_vs_{b}'].value = self.delta_v[ts]
                self.param[f'us_{b}'].value = self.u[ts]

                vpar_inds = self.vpar_inds[i, :t].nonzero()[0]
                ts = np.concatenate([
                    vpar_inds[-k:],
                    self.rng.choice(len(vpar_inds) - k, size=self.nsamples-k, replace=False)
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
            self.log.write(f'{indent} CBC prob.status = {prob.status}')
            if prob.status == 'infeasible':
                import pdb
                pdb.set_trace()
        self.X_cache = np.array(X.value)  # make a copy
        make_pd_and_pos(self.X_cache)
        self.is_cached = True

        # check slack variable
        if slack_w.value > 0:
            self.log.write(f'{indent} CBC slack: {slack_w.value:.3f}')

        # check whether constraints are satisfied for latest time step
        satisfied, msg = self._check_newest_obs(t)
        if not satisfied:
            self.log.write(f'{indent} CBC post opt: {msg}')

        return np.array(self.X_cache)  # return a copy


# class CBCProjectionWithNoise(CBCProjection):
#     def __init__(self, eta: float, n: int, T: int, nsamples: int,
#                  alpha: float, v: np.ndarray, X_init: np.ndarray | None = None,
#                  X_true: np.ndarray | None = None):
#         """
#         Same args as CBCProjection. However, here, we interpret eta as an upper
#         limit on true noise.
#         """
#         super().__init__(eta, n, T, nsamples, alpha, v, X_init, X_true)
#         self.var_eta = cp.Variable(nonneg=True)
#         self.eta_cache = 0

#     def select(self) -> tuple[np.ndarray, float]:
#         """
#         When select() is called, we have seen self.t observations.
#         """
#         if self.is_cached:
#             return self.X_cache, self.eta_cache

#         t = self.t
#         assert t >= 1

#         # be lazy if self.X_cache already satisfies the newest obs.
#         est_noise = self.delta_v[:, t-1] - self.X_cache @ self.u[:, t-1]
#         # tqdm.write(f'est_noise: {np.max(np.abs(est_noise)):.3f}')
#         if np.max(np.abs(est_noise)) <= self.eta_cache:
#             # buf = self.eta - np.max(np.abs(est_noise))
#             # self.lazy_buffer.append(buf)
#             # tqdm.write('being lazy')
#             self.is_cached = True
#             return self.X_cache, self.eta_cache

#         n = self.n

#         # optimization variables
#         X = self.var_X
#         slack = self.var_slack
#         eta = self.var_eta

#         ub = self.var_eta  # * np.ones([n, 1])
#         lb = -ub

#         # when t < self.nsamples, create a brand-new cp.Problem
#         if t < self.nsamples:
#             us = self.us[:, :t]
#             delta_vs = self.delta_v[:, :t]

#             diffs = delta_vs - X @ us
#             constrs = [X >= 0, lb <= diffs, diffs <= ub, eta <= self.eta]
#             # constrs = [X >= 0, lb + slack <= diffs, diffs <= ub - slack,
#             #            eta <= self.eta]

#             obj = cp.Minimize(cp_triangle_norm_sq(X - self.X_cache)
#                               + 1e3 * eta**2)
#             # obj = cp.Minimize(cp_triangle_norm_sq(X - self.X_cache)
#             #                   + (eta - self.eta_cache)**2 + eta**2)
#             # obj = cp.Minimize(cp_triangle_norm_sq(X - self.X_cache)
#             #                   + (eta - self.eta_cache)**2
#             #                   - self.alpha * slack)
#             prob = cp.Problem(objective=obj, constraints=constrs)
#             prob.solve()

#         # when t >= self.nsamples, compile a fixed-size optimization problem
#         else:
#             if self.prob is None:
#                 Xprev = cp.Parameter([n, n], nonneg=True, name='Xprev')
#                 etaprev = cp.Parameter(nonneg=True, name='eta')
#                 us = cp.Parameter([n, self.nsamples], name='us')
#                 delta_vs = cp.Parameter([n, self.nsamples], name='delta_vs')

#                 diffs = delta_vs - X @ us
#                 # constrs = [X >= 0, lb <= diffs, diffs <= ub, eta <= self.eta]
#                 constrs = [X >= 0, lb + slack <= diffs, diffs <= ub - slack,
#                            etaprev <= eta, eta <= self.eta]

#                 obj = cp.Minimize(cp_triangle_norm_sq(X - Xprev)
#                                   + 3e2 * eta**2
#                                   - self.alpha * slack)
#                 # obj = cp.Minimize(cp_triangle_norm_sq(X - Xprev)
#                 #                   + (eta - etaprev)**2)
#                 # obj = cp.Minimize(cp_triangle_norm_sq(X - Xprev)
#                 #                   + (eta - etaprev)**2
#                 #                   - self.alpha * slack)
#                 self.prob = cp.Problem(objective=obj, constraints=constrs)

#                 # if CBC problem is DPP, then it can be compiled for speedup
#                 # - see https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming  # noqa
#                 tqdm.write(f'CBC prob is DPP?: {self.prob.is_dcp(dpp=True)}')

#                 self.param_Xprev = Xprev
#                 self.param_etaprev = etaprev
#                 self.param_us = us
#                 self.param_delta_vs = delta_vs

#             prob = self.prob

#             # perform random sampling
#             # - use the most recent k (<=5) time steps
#             # - then sample additional previous time steps for 20 total
#             k = min(self.nsamples, 5)
#             sample_probs = np.linalg.norm(
#                 self.delta_v[:, :t-k] - self.X_cache @ self.us[:, :t-k],
#                 axis=0)
#             sample_probs /= np.sum(sample_probs)
#             ts = np.concatenate([
#                 np.arange(t-k, t),
#                 rng.choice(t-k, size=self.nsamples-k, replace=False,
#                            p=sample_probs)
#             ])
#             self.param_us.value = self.us[:, ts]
#             self.param_delta_vs.value = self.delta_v[:, ts]

#             self.param_Xprev.value = self.X_cache
#             self.param_etaprev.value = self.eta_cache
#             prob.solve(warm_start=True)

#         if prob.status != 'optimal':
#             tqdm.write(f'CBC prob.status = {prob.status}')
#             if prob.status == 'infeasible':
#                 import pdb
#                 pdb.set_trace()
#         self.X_cache = np.array(X.value)  # make a copy
#         self.eta_cache = float(eta.value)  # make a copy

#         # Force symmetry, even if all-close. But only print error message if
#         # not all-close.
#         if not np.allclose(self.X_cache, self.X_cache.T):
#             max_diff = np.max(np.abs(self.X_cache - self.X_cache.T))
#             tqdm.write(f'optimal X not symmetric. ||X-X.T||_max = {max_diff}'
#                        ' - making symmetric')
#         self.X_cache = (self.X_cache + self.X_cache.T) / 2

#         # check for PSD
#         w, V = np.linalg.eigh(self.X_cache)
#         if np.any(w < 0):
#             tqdm.write(f'optimal X not PSD. smallest eigenvalue = {np.min(w)}'
#                        ' - setting neg eigenvalues to 0')
#             w[w < 0] = 0
#             self.X_cache = (V * w) @ V.T

#         if np.any(self.X_cache < 0):
#             tqdm.write(f'optimal X has neg values. min={np.min(self.X_cache)}'
#                        ' - applying ReLu')
#             self.X_cache = np.maximum(0, self.X_cache)

#         self.is_cached = True
#         return (self.X_cache, self.eta_cache)
