"""Convex body chasing via projection."""
from __future__ import annotations

from collections.abc import Callable, Sequence
import io

import cvxpy as cp
import numpy as np
from tqdm.auto import tqdm

from cbc.base import CBCBase, cp_triangle_norm_sq
from network_utils import make_pd_and_pos
from utils import solve_prob


def check_informative(t: int, b: np.ndarray, c: np.ndarray,
                      useful: np.ndarray) -> None:
    """Checks whether b[t], c[t] are useful.

    Args
    - t: int
    - b, c: np.ndarray, shape [T, n]
    - useful: np.ndarray, shape [2, T], boolean indexing vector
        - 1st row is for lower bound, 2nd row is for upper bound
    """
    # manage constraints of the form: d <= b - X c
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


def sample_ts(rng: np.random.Generator, valid: np.ndarray | Sequence[int],
              total: int, num_recent: int, num_update: int,
              ts_updated: Sequence[int] | None = None
              ) -> np.ndarray:
    """Samples time steps based on given criteria.

    Samples:
    1. num_recent most recent steps
    2. num_update steps that required model updating
    3. (total - num_recent - num_update) steps randomly

    Args
    - rng: numpy random number generator
    - valid: list of time steps to choose from
    - total: total number of time steps to sample
    - num_recent: include num_recent most recent time steps
    - num_update: include num_update time steps that required model updates
    - ts_updated: list of time steps where model required updates

    Returns: list of time steps
    """
    recent_ts = valid[-num_recent:]

    if num_update == 0:
        rand_ts = rng.choice(
            valid[:-num_recent], size=total - num_recent, replace=False)
        ts = np.concatenate([recent_ts, rand_ts])

    else:
        assert ts_updated is not None

        valid_update_ts = np.setdiff1d(ts_updated, recent_ts)
        update_ts = rng.choice(
            valid_update_ts, size=min(num_update, len(valid_update_ts)),
            replace=False)
        valid_rand_ts = np.setdiff1d(valid[:-num_recent], update_ts)
        rand_ts = rng.choice(
            valid_rand_ts, size=total - num_recent - len(update_ts),
            replace=False)
        ts = np.concatenate([recent_ts, update_ts, rand_ts])

    return ts


class CBCProjection(CBCBase):
    """Finds the set of X that is consistent with the observed data. Assumes
    that noise bound (eta) is known.
    """
    def __init__(self, n: int, T: int, X_init: np.ndarray, v: np.ndarray,
                 gen_X_set: Callable[[cp.Variable], list[cp.Constraint]],
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
            set to 0 to turn off slack variable
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

        self.w_inds = np.zeros([2, T-1], dtype=bool)  # whether each (u(t), Δv(t)) is useful
        self.vpar_inds = np.zeros([2, T], dtype=bool)  # whether each (v(t), q(t)) is useful
        self.w_inds[:, 0] = True
        self.vpar_inds[:, 1] = True
        self.ts_updated: list[int] = []

        self.var_slack_w = cp.Variable(nonneg=True) if alpha > 0 else cp.Constant(0.)
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

        constrs = self.X_set[:]  # make a shallow copy
        obs = self.obs_nodes
        self.param = {}

        Xprev = cp.Parameter((n, n), PSD=True, name='Xprev')
        for b in ['lb', 'ub']:
            vs = cp.Parameter((self.nsamples, n), name=f'vs_{b}')
            Δvs = cp.Parameter((self.nsamples, n), name=f'Δvs_{b}')
            us = cp.Parameter((self.nsamples, n), name=f'us_{b}')
            qs = cp.Parameter((self.nsamples, n), name=f'qs_{b}')

            ŵs = Δvs - us @ X
            vpar_hats = vs - qs @ X

            if b == 'lb':
                constrs.extend([
                    lb - slack_w <= ŵs,
                    self.Vpar_min[None, obs] <= vpar_hats[:, obs]
                ])
            else:
                constrs.extend([
                    ŵs <= ub + slack_w,
                    vpar_hats[:, obs] <= self.Vpar_max[None, obs]
                ])

            self.param[f'vs_{b}'] = vs
            self.param[f'Δvs_{b}'] = Δvs
            self.param[f'us_{b}'] = us
            self.param[f'qs_{b}'] = qs
        self.param['Xprev'] = Xprev

        # constrs = self.X_set[:]  # make a shallow copy + [
        #     lb - slack_w <= ŵs, ŵs <= ub + slack_w,
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
        # self.param_Δvs = Δvs
        # self.param_us = us
        # self.param_qs = qs

    def add_obs(self, t: int) -> None:
        """
        Args
        - t: int, current time step (>=1), v[t] and q[t] have just been updated
        """
        # update self.u and self.Δv
        super().add_obs(t)

        if self.is_cached:
            satisfied, msg = self._check_newest_obs(t)
            if not satisfied:
                self.is_cached = False
                self.ts_updated.append(t)
                self.log.write(f't = {t:6d}, CBC pre opt: {msg}')

        if t >= 2:
            check_informative(t=t-1, b=self.Δv, c=self.u, useful=self.w_inds)
            check_informative(t=t, b=self.v, c=self.q, useful=self.vpar_inds)

        if t % 500 == 0:
            num_w_inds = tuple(np.sum(self.w_inds[:t], axis=1))
            num_vpar_inds = tuple(np.sum(self.vpar_inds[:t+1], axis=1))
            self.log.write(f'active constraints - w: {num_w_inds}/{t}, vpar: {num_vpar_inds}/{t}')

    def _check_newest_obs(self, t: int, X_test: np.ndarray | None = None) -> tuple[bool, str]:
        """Checks whether self.X_cache (or X_test, if given) satisfies the
        newest observation: (v[t], q[t], u[t-1], Δv[t-1])

        Even when CVXPY solves SEL to optimality, empirically it may still have
        up to 0.05 of constraint violation, so we allow for that here.

        Returns
        - satisfied: bool, whether self.X_cache satisfies the newest observation
        - msg: str, (if not satisfied) describes which constraints are not satisfied,
            (if satisfied) is empty string ''
        """
        X = self.X_cache if X_test is None else X_test

        obs = self.obs_nodes
        w_hat = self.Δv[t-1] - self.u[t-1] @ X
        vpar_hat = self.v[t] - self.q[t] @ X
        w_hat_norm = np.max(np.abs(w_hat[obs]))

        vpar_lower_violation = np.max(self.Vpar_min[obs] - vpar_hat[obs])
        vpar_upper_violation = np.max(vpar_hat[obs] - self.Vpar_max[obs])

        msgs = []
        if w_hat_norm > self.eta:
            msgs.append(f'‖ŵ(t)‖∞: {w_hat_norm:.3f}')
        if vpar_lower_violation > 0.05:
            msgs.append(f'max(vpar_min - vpar_hat): {vpar_lower_violation:.3f}')
        if vpar_upper_violation > 0.05:
            msgs.append(f'max(vpar_hat - vpar_max): {vpar_upper_violation:.3f}')
        satisfied = (len(msgs) == 0)
        msg = ', '.join(msgs)
        return satisfied, msg

    def select(self, t: int) -> np.ndarray:
        """Selects the closest consistent model.

        We have seen t observations. That is, we have values for:
          v(0), ...,   v(t)    # recall:   v(t) = vs[t]
        q^c(0), ..., q^c(t)    # recall: q^c(t) = qs[t]
          u(0), ...,   u(t-1)  # recall:   u(t) = us[t]
         Δv(0), ...,  Δv(t-1)  # recall:  Δv(t) = Δvs[t]

        It is possible that t=0, meaning we haven't seen any observations yet.
        (We have v(0) and q^c(0), but not u(0) or Δv(0).) In this case, our
        X_init should be cached, and we will return that.

        Args
        - t: int, current time step (>=0)

        Returns:
        - Xhat: np.ndarray, shape [n, n], consistent model
        """
        # be lazy if self.X_cache already satisfies the newest obs.
        if self.is_cached:
            return self.X_cache

        indent = ' ' * 11

        # optimization variables
        # - If assumptions 1, 2, and the first part of 3
        #     ($\forall t: \vpar(t) \in \Vpar$) are satisfied, we don't need
        #     need a slack variable in SEL (the CBC algorithm). However, in
        #     practice, it is often difficult to check these assumptions, so we
        #     include a slack variable in case of infeasibility.
        X = self.var_X
        slack_w = self.var_slack_w

        # when t < self.nsamples
        if t < self.nsamples:
            for b in ['lb', 'ub']:
                self.param[f'vs_{b}'].value = np.tile(self.Vpar_min, [self.nsamples, 1])
                self.param[f'vs_{b}'].value[:t] = self.v[1:1+t]
                self.param[f'Δvs_{b}'].value = self.Δv[:self.nsamples]
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
                ts = sample_ts(self.rng, w_inds, total=self.nsamples,
                               num_recent=k, num_update=0)

                self.param[f'Δvs_{b}'].value = self.Δv[ts]
                self.param[f'us_{b}'].value = self.u[ts]

                vpar_inds = self.vpar_inds[i, :t+1].nonzero()[0]
                ts = sample_ts(self.rng, vpar_inds, total=self.nsamples,
                               num_recent=k, num_update=0)

                self.param[f'vs_{b}'].value = self.v[ts]
                self.param[f'qs_{b}'].value = self.q[ts]

            # self.param_vs.value = self.v[ts+1]
            # self.param_Δvs.value = self.Δv[ts]
            # self.param_us.value = self.us[ts]
            # self.param_qs.value = self.q[ts+1]

        # self.param_Xprev.value = self.X_cache
        self.param['Xprev'].value = self.X_cache

        solve_prob(self.prob, log=self.log, name='CBC', indent=indent)

        self.X_cache = np.array(X.value)  # make a copy
        make_pd_and_pos(self.X_cache)
        self.is_cached = True

        # check slack variable
        if slack_w.value > 0:
            self.log.write(f'{indent} CBC slack: {slack_w.value:.3f}')

        # check whether constraints are satisfied for latest time step
        # print('check if the new model is good.')
        satisfied, msg = self._check_newest_obs(t)
        if not satisfied:
            self.log.write(f'{indent} CBC post opt: {msg}')

        return np.array(self.X_cache)  # return a copy


class CBCProjectionWithNoise(CBCProjection):
    def __init__(self, n: int, T: int, X_init: np.ndarray, v: np.ndarray,
                 gen_X_set: Callable[[cp.Variable], list[cp.Constraint]],
                 eta: float, nsamples: int, δ: float,
                 Vpar: tuple[np.ndarray, np.ndarray],
                 X_true: np.ndarray,
                 obs_nodes: Sequence[int] | None = None,
                 log: tqdm | io.TextIOBase | None = None, seed: int = 123):
        """
        Args:
        - δ: float, weight of noise term in CBC norm
        - all other args are the same as CBCProjection. However, here, we
            interpret eta as an upper limit on true noise. We also remove the
            slack variable, which should be unnecessary as long as eta is set
            large enough.
        """
        self.var_eta = cp.Variable(nonneg=True)
        self.eta_max = eta  # upper limit on true noise
        self.δ = δ
        alpha = 0
        super().__init__(n, T, X_init, v, gen_X_set, eta, nsamples, alpha,
                         Vpar, X_true, obs_nodes, log, seed)
        self.eta = 0  # cached value

    def _setup_prob(self) -> None:
        """Defines self.prob as the projection of Xprev into the consistent set.
        """
        n = self.n
        ub = self.var_eta
        lb = -ub

        # optimization variables
        X = self.var_X
        var_eta = self.var_eta

        constrs = self.X_set[:]  # make a shallow copy
        constrs.append(var_eta <= self.eta_max)
        obs = self.obs_nodes
        self.param = {}

        Xprev = cp.Parameter((n, n), PSD=True, name='Xprev')
        etaprev = cp.Parameter(nonneg=True, name='etaprev')
        for b in ['lb', 'ub']:
            vs = cp.Parameter((self.nsamples, n), name=f'vs_{b}')
            Δvs = cp.Parameter((self.nsamples, n), name=f'Δvs_{b}')
            us = cp.Parameter((self.nsamples, n), name=f'us_{b}')
            qs = cp.Parameter((self.nsamples, n), name=f'qs_{b}')

            ŵs = Δvs - us @ X
            vpar_hats = vs - qs @ X

            if b == 'lb':
                constrs.extend([
                    lb <= ŵs,
                    self.Vpar_min[None, obs] <= vpar_hats[:, obs]
                ])
            else:
                constrs.extend([
                    ŵs <= ub,
                    vpar_hats[:, obs] <= self.Vpar_max[None, obs]
                ])

            self.param[f'vs_{b}'] = vs
            self.param[f'Δvs_{b}'] = Δvs
            self.param[f'us_{b}'] = us
            self.param[f'qs_{b}'] = qs
        self.param['Xprev'] = Xprev
        self.param['etaprev'] = etaprev

        obj = cp.Minimize(cp_triangle_norm_sq(X-Xprev)
                          + (self.δ * (var_eta - etaprev))**2)
        self.prob = cp.Problem(objective=obj, constraints=constrs)

        # if cp.Problem is DPP, then it can be compiled for speedup
        # - http://cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming  # noqa
        self.log.write(f'CBC prob is DPP?: {self.prob.is_dcp(dpp=True)}')

    def select(self, t: int) -> tuple[np.ndarray, float]:  # type: ignore
        """
        When select() is called, we have seen t observations.
        """
        # be lazy if (self.X_cache, self.eta) already satisfies the newest obs.
        if self.is_cached:
            return self.X_cache, self.eta

        indent = ' ' * 11

        # optimization variables
        # - If assumptions 1, 2, and the first part of 3
        #     ($\forall t: \vpar(t) \in \Vpar$) are satisfied, we don't need
        #     need a slack variable in SEL (the CBC algorithm). However, in
        #     practice, it is often difficult to check these assumptions, so we
        #     include a slack variable in case of infeasibility.
        X = self.var_X
        var_eta = self.var_eta

        # when t < self.nsamples
        if t < self.nsamples:
            for b in ['lb', 'ub']:
                self.param[f'vs_{b}'].value = np.tile(self.Vpar_min, [self.nsamples, 1])
                self.param[f'vs_{b}'].value[:t] = self.v[1:1+t]
                self.param[f'Δvs_{b}'].value = self.Δv[:self.nsamples]
                self.param[f'us_{b}'].value = self.u[:self.nsamples]
                self.param[f'qs_{b}'].value = self.q[1:1+self.nsamples]

        # when t >= self.nsamples
        else:
            # perform random sampling
            # - use the most recent k time steps  [t-k, ..., t-1]
            # - then sample additional previous time steps for self.nsamples total
            #   [0, ..., t-k-1]
            k = min(self.nsamples, 20)

            for i, b in enumerate(['lb', 'ub']):
                w_inds = self.w_inds[i, :t].nonzero()[0]
                ts = sample_ts(self.rng, w_inds, total=self.nsamples,
                               num_recent=k, num_update=0)

                self.param[f'Δvs_{b}'].value = self.Δv[ts]
                self.param[f'us_{b}'].value = self.u[ts]

                vpar_inds = self.vpar_inds[i, :t+1].nonzero()[0]
                ts = sample_ts(self.rng, vpar_inds, total=self.nsamples,
                               num_recent=k, num_update=0)

                self.param[f'vs_{b}'].value = self.v[ts]
                self.param[f'qs_{b}'].value = self.q[ts]

        self.param['Xprev'].value = self.X_cache
        self.param['etaprev'].value = self.eta

        solve_prob(self.prob, log=self.log, name='CBC', indent=indent)

        self.X_cache = np.array(X.value)  # make a copy
        self.eta = float(var_eta.value)  # make a copy
        make_pd_and_pos(self.X_cache)
        self.is_cached = True

        # check whether constraints are satisfied for latest time step
        # print('check if the new model is good.')
        satisfied, msg = self._check_newest_obs(t)
        if not satisfied:
            self.log.write(f'{indent} CBC post opt: {msg}')

        return np.array(self.X_cache), self.eta  # return a copy
