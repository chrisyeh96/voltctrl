"""Convex body chasing via projection."""
from __future__ import annotations

from collections.abc import Callable, Sequence
import io

import cvxpy as cp
import numpy as np
from tqdm.auto import tqdm

from cbc.base import CBCBase, CBCInfeasibleError, cp_triangle_norm_sq
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
                 prune_constraints: bool = False,
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
        - prune_constraints: whether to attempt to remove constraints that are
            already handled by other constraints
        - seed: int, random seed
        """
        assert seed is not None
        super().__init__(n=n, T=T, X_init=X_init, v=v, gen_X_set=gen_X_set,
                         X_true=X_true, obs_nodes=obs_nodes, log=log)
        self.is_cached = True

        self.eta = eta
        self.nsamples = nsamples
        self.alpha = alpha

        self.prune_constraints = prune_constraints
        if prune_constraints:
            self.w_inds = np.zeros([2, T-1], dtype=bool)  # whether each (u(t), Δv(t)) is useful
            self.vpar_inds = np.zeros([2, T], dtype=bool)  # whether each (v(t), q(t)) is useful
            self.w_inds[:, 0] = True
            self.vpar_inds[:, 1] = True

        self.var_slack_w = cp.Variable(nonneg=True) if alpha > 0 else cp.Constant(0.)
        self.Vpar_min, self.Vpar_max = Vpar

        self._setup_prob()
        self.rng = np.random.default_rng(seed)

    def _setup_prob(self) -> None:
        """Defines self.prob as the projection of Xprev into consistent set.
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
        self.param['Xprev'] = Xprev

        if self.prune_constraints:
            for b in ['lb', 'ub']:
                vs = cp.Parameter((self.nsamples, n), name=f'vs_{b}')
                Δvs = cp.Parameter((self.nsamples, n), name=f'Δvs_{b}')
                us = cp.Parameter((self.nsamples, n), name=f'us_{b}')
                qs = cp.Parameter((self.nsamples, n), name=f'qs_{b}')

                ŵs = Δvs - us @ X
                vpar_hats = vs - qs @ X

                if b == 'lb':
                    constrs.extend([
                        lb - slack_w <= ŵs[:, obs],
                        self.Vpar_min[None, obs] <= vpar_hats[:, obs]
                    ])
                else:
                    constrs.extend([
                        ŵs[:, obs] <= ub + slack_w,
                        vpar_hats[:, obs] <= self.Vpar_max[None, obs]
                    ])

                self.param[f'vs_{b}'] = vs
                self.param[f'Δvs_{b}'] = Δvs
                self.param[f'us_{b}'] = us
                self.param[f'qs_{b}'] = qs
        else:
            vs = cp.Parameter((self.nsamples, n), name='vs')
            Δvs = cp.Parameter((self.nsamples, n), name='Δvs')
            us = cp.Parameter((self.nsamples, n), name='us')
            qs = cp.Parameter((self.nsamples, n), name='qs')

            ŵs = Δvs - us @ X
            vpar_hats = vs - qs @ X

            constrs.extend([
                lb - slack_w <= ŵs[:, obs], ŵs[:, obs] <= ub + slack_w,
                self.Vpar_min[None, obs] <= vpar_hats[:, obs],
                vpar_hats[:, obs] <= self.Vpar_max[None, obs]
            ])

            self.param['vs'] = vs
            self.param['Δvs'] = Δvs
            self.param['us'] = us
            self.param['qs'] = qs

        obj = cp.Minimize(cp_triangle_norm_sq(X-Xprev) + self.alpha * slack_w)
        self.prob = cp.Problem(objective=obj, constraints=constrs)

        # if cp.Problem is DPP, then it can be compiled for speedup
        # - http://cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming  # noqa
        self.log.write(f'CBC prob is DPP?: {self.prob.is_dcp(dpp=True)}')

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

        if self.prune_constraints:
            if t >= 2:
                check_informative(t=t-1, b=self.Δv, c=self.u, useful=self.w_inds)
                check_informative(t=t, b=self.v, c=self.q, useful=self.vpar_inds)

            if t % 500 == 0:
                num_w_inds = tuple(np.sum(self.w_inds[:t], axis=1))
                num_vpar_inds = tuple(np.sum(self.vpar_inds[:t+1], axis=1))
                self.log.write(f'active constraints - w: {num_w_inds}/{t}, '
                               f'vpar: {num_vpar_inds}/{t}')

    def _check_newest_obs(self, t: int, X_test: np.ndarray | None = None
                          ) -> tuple[bool, str]:
        """Checks whether self.X_cache (or X_test, if given) satisfies the
        newest observation: (v[t], q[t], u[t-1], Δv[t-1])

        Even when CVXPY solves SEL to optimality, empirically it may still have
        up to 0.05 of constraint violation, so we allow for that here.

        Returns
        - satisfied: bool, whether self.X_cache (or X_test, if given) satisfies
            the newest observation
        - msg: str, (if not satisfied) describes which constraints are violated
            (if satisfied) is empty string ''
        """
        X = self.X_cache if X_test is None else X_test

        obs = self.obs_nodes
        ŵ = self.Δv[t-1] - self.u[t-1] @ X
        vpar_hat = self.v[t] - self.q[t] @ X
        ŵ_norm = np.max(np.abs(ŵ[obs]))

        vpar_lower_violation = np.max(self.Vpar_min[obs] - vpar_hat[obs])
        vpar_upper_violation = np.max(vpar_hat[obs] - self.Vpar_max[obs])

        msgs = []
        if ŵ_norm > self.eta:
            msgs.append(f'‖ŵ(t)‖∞: {ŵ_norm:.3f}')
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
            if self.prune_constraints:
                for b in ['lb', 'ub']:
                    self.param[f'vs_{b}'].value = np.tile(self.Vpar_min, [self.nsamples, 1])
                    self.param[f'vs_{b}'].value[:t] = self.v[1:1+t]
                    self.param[f'Δvs_{b}'].value = self.Δv[:self.nsamples]
                    self.param[f'us_{b}'].value = self.u[:self.nsamples]
                    self.param[f'qs_{b}'].value = self.q[1:1+self.nsamples]
            else:
                self.param['vs'].value = np.tile(self.Vpar_min, [self.nsamples, 1])
                self.param['vs'].value[:t] = self.v[1:1+t]
                self.param['Δvs'].value = self.Δv[:self.nsamples]
                self.param['us'].value = self.u[:self.nsamples]
                self.param['qs'].value = self.q[1:1+self.nsamples]

        # when t >= self.nsamples
        else:
            # always include the most recent k time steps  [t-k, ..., t-1]
            k = min(self.nsamples, 20)

            if self.prune_constraints:
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

            else:
                ts = sample_ts(self.rng, np.arange(t), total=self.nsamples,
                               num_recent=k, num_update=0)
                self.param['Δvs'].value = self.Δv[ts]
                self.param['us'].value = self.u[ts]
                self.param['vs'].value = self.v[ts+1]
                self.param['qs'].value = self.q[ts+1]

        self.param['Xprev'].value = self.X_cache

        solve_prob(self.prob, log=self.log, name='CBC', indent=indent)
        if self.prob.status == 'infeasible':
            raise CBCInfeasibleError

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

    def is_consistent(self, t: int, X: np.ndarray) -> tuple[bool, bool]:
        """Returns whether parameter estimate X is consistent with the
        full set of observations up to time t.

        Assumes that X is in 𝒳. Therefore, we only have to check the data-driven
        constraints that define the consistent set.

        Returns:
        - consistent: whether X is consistent
        - consistent_05: allows for 0.05 in vpar constraint violation, because
            CVXPY empirically may still have up to 0.05 of constraint violation,
            even when it solves SEL to "optimality"
        """
        if t == 0:
            return True, True

        obs = self.obs_nodes
        ŵ = self.Δv[:t] - self.u[:t] @ X  # shape [t, n]
        vpar_hat = self.v[:t+1] - self.q[:t+1] @ X  # shape [t+1, n]
        ŵ_norm = np.max(np.abs(ŵ[:, obs]))

        vpar_lower_violation = np.max(self.Vpar_min[obs] - vpar_hat[:, obs])
        vpar_upper_violation = np.max(vpar_hat[:, obs] - self.Vpar_max[obs])

        consistent = True
        consistent_05 = True  # allows for 0.05 buffer in vpar violation

        if ŵ_norm > self.eta:
            consistent = False
            consistent_05 = False

        if vpar_lower_violation > 0 or vpar_upper_violation > 0:
            consistent = False
        if vpar_lower_violation > 0.05 or vpar_upper_violation > 0.05:
            consistent_05 = False

        return consistent, consistent_05


class CBCProjectionWithNoise(CBCProjection):
    def __init__(self, n: int, T: int, X_init: np.ndarray, eta_init: float,
                 v: np.ndarray,
                 gen_X_set: Callable[[cp.Variable], list[cp.Constraint]],
                 eta: float, nsamples: int, δ: float,
                 Vpar: tuple[np.ndarray, np.ndarray],
                 X_true: np.ndarray,
                 obs_nodes: Sequence[int] | None = None,
                 prune_constraints: bool = False,
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
                         Vpar, X_true, obs_nodes, prune_constraints, log, seed)
        self.eta = eta_init  # cached value

    def _setup_prob(self) -> None:
        """Defines self.prob as the projection of Xprev into consistent set.
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
        self.param['Xprev'] = Xprev
        self.param['etaprev'] = etaprev

        if self.prune_constraints:
            for b in ['lb', 'ub']:
                vs = cp.Parameter((self.nsamples, n), name=f'vs_{b}')
                Δvs = cp.Parameter((self.nsamples, n), name=f'Δvs_{b}')
                us = cp.Parameter((self.nsamples, n), name=f'us_{b}')
                qs = cp.Parameter((self.nsamples, n), name=f'qs_{b}')

                ŵs = Δvs - us @ X
                vpar_hats = vs - qs @ X

                if b == 'lb':
                    constrs.extend([
                        lb <= ŵs[:, obs],
                        self.Vpar_min[None, obs] <= vpar_hats[:, obs]
                    ])
                else:
                    constrs.extend([
                        ŵs[:, obs] <= ub,
                        vpar_hats[:, obs] <= self.Vpar_max[None, obs]
                    ])

                self.param[f'vs_{b}'] = vs
                self.param[f'Δvs_{b}'] = Δvs
                self.param[f'us_{b}'] = us
                self.param[f'qs_{b}'] = qs

        else:
            vs = cp.Parameter((self.nsamples, n), name='vs')
            Δvs = cp.Parameter((self.nsamples, n), name='Δvs')
            us = cp.Parameter((self.nsamples, n), name='us')
            qs = cp.Parameter((self.nsamples, n), name='qs')

            ŵs = Δvs - us @ X
            vpar_hats = vs - qs @ X

            constrs.extend([
                lb <= ŵs[:, obs], ŵs[:, obs] <= ub,
                self.Vpar_min[None, obs] <= vpar_hats[:, obs],
                vpar_hats[:, obs] <= self.Vpar_max[None, obs]
            ])

            self.param['vs'] = vs
            self.param['Δvs'] = Δvs
            self.param['us'] = us
            self.param['qs'] = qs

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
            if self.prune_constraints:
                for b in ['lb', 'ub']:
                    self.param[f'vs_{b}'].value = np.tile(self.Vpar_min, [self.nsamples, 1])
                    self.param[f'vs_{b}'].value[:t] = self.v[1:1+t]
                    self.param[f'Δvs_{b}'].value = self.Δv[:self.nsamples]
                    self.param[f'us_{b}'].value = self.u[:self.nsamples]
                    self.param[f'qs_{b}'].value = self.q[1:1+self.nsamples]
            else:
                self.param['vs'].value = np.tile(self.Vpar_min, [self.nsamples, 1])
                self.param['vs'].value[:t] = self.v[1:1+t]
                self.param['Δvs'].value = self.Δv[:self.nsamples]
                self.param['us'].value = self.u[:self.nsamples]
                self.param['qs'].value = self.q[1:1+self.nsamples]

        # when t >= self.nsamples
        else:
            # always include the most recent k time steps  [t-k, ..., t-1]
            k = min(self.nsamples, 20)

            if self.prune_constraints:
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

            else:
                ts = sample_ts(self.rng, np.arange(t), total=self.nsamples,
                               num_recent=k, num_update=0)
                self.param['Δvs'].value = self.Δv[ts]
                self.param['us'].value = self.u[ts]
                self.param['vs'].value = self.v[ts+1]
                self.param['qs'].value = self.q[ts+1]

        self.param['Xprev'].value = self.X_cache
        self.param['etaprev'].value = self.eta

        solve_prob(self.prob, log=self.log, name='CBC', indent=indent)
        if self.prob.status == 'infeasible':
            raise CBCInfeasibleError

        self.X_cache = np.array(X.value)  # make a copy
        self.eta = float(var_eta.value)  # make a copy
        make_pd_and_pos(self.X_cache)
        self.is_cached = True

        # check whether constraints are satisfied for latest time step
        satisfied, msg = self._check_newest_obs(t)
        if not satisfied:
            self.log.write(f'{indent} CBC post opt: {msg}')

        return np.array(self.X_cache), self.eta  # return a copy

    def is_consistent(self, t: int, X: np.ndarray, eta: float | None = None) -> tuple[bool, bool]:
        """Returns whether parameter estimates (X, eta) are consistent with the
        full set of observations up to time t.

        Assumes that X is in 𝒳, and eta is in [0, etamax]. Therefore, we only
        have to check the data-driven constraints that define the consistent set.

        Returns:
        - consistent: whether (X, eta) are consistent
        - consistent_05: allows for 0.05 in vpar constraint violation, because
            CVXPY empirically may still have up to 0.05 of constraint violation,
            even when it solves SEL to "optimality"
        """
        if t == 0:
            return True, True

        if eta is None:
            eta = self.eta

        obs = self.obs_nodes
        ŵ = self.Δv[:t] - self.u[:t] @ X  # shape [t, n]
        vpar_hat = self.v[:t+1] - self.q[:t+1] @ X  # shape [t+1, n]
        ŵ_norm = np.max(np.abs(ŵ[:, obs]))

        vpar_lower_violation = np.max(self.Vpar_min[obs] - vpar_hat[:, obs])
        vpar_upper_violation = np.max(vpar_hat[:, obs] - self.Vpar_max[obs])

        consistent = True
        consistent_05 = True  # allows for 0.05 buffer in vpar violation

        if ŵ_norm > eta:
            consistent = False
            consistent_05 = False

        if vpar_lower_violation > 0 or vpar_upper_violation > 0:
            consistent = False
        if vpar_lower_violation > 0.05 or vpar_upper_violation > 0.05:
            consistent_05 = False

        return consistent, consistent_05
