"""Convex body chasing via Steiner point."""
from __future__ import annotations

from collections.abc import Callable, Sequence
import io

import cvxpy as cp
import numpy as np
from tqdm.auto import tqdm

from cbc.base import CBCBase
from cbc.projection import check_informative, sample_ts
from network_utils import make_pd_and_pos
from utils import solve_prob


class CBCSteiner(CBCBase):
    """Finds the set of X that is consistent with the observed data. Assumes
    that noise bound (eta) is known.
    """
    def __init__(self, n: int, T: int, X_init: np.ndarray, v: np.ndarray,
                 gen_X_set: Callable[[cp.Variable], list[cp.Constraint]],
                 eta: float, nsamples: int, nsamples_steiner: int,
                 Vpar: tuple[np.ndarray, np.ndarray],
                 X_true: np.ndarray, obs_nodes: Sequence[int] | None = None,
                 prune_constraints: bool = False,
                 log: tqdm | io.TextIOBase | None = None, seed: int = 123):
        """
        Args
        - see CBCBase for descriptions of other parameters
        - eta: float, noise bound
        - nsamples: int, # of observations to use for defining the convex set
        - nsamples_steiner: int, # of random directions to use for estimating
            the Steiner point integral
        - Vpar: tuple (Vpar_min, Vpar_max), box description of Vpar
            - each Vpar_* is a np.array of shape [n]
        - seed: int, random seed
        """
        super().__init__(n=n, T=T, X_init=X_init, v=v, gen_X_set=gen_X_set,
                         X_true=X_true, obs_nodes=obs_nodes, log=log)
        self.dim = n * (n+1) // 2
        self.is_cached = True

        self.eta = eta
        self.nsamples = nsamples
        self.nsamples_steiner = nsamples_steiner

        self.prune_constraints = prune_constraints
        if prune_constraints:
            self.w_inds = np.zeros([2, T-1], dtype=bool)  # whether each (u(t), Δv(t)) is useful
            self.vpar_inds = np.zeros([2, T], dtype=bool)  # whether each (v(t), q(t)) is useful
            self.w_inds[:, 0] = True
            self.vpar_inds[:, 1] = True

        self.var_slack_w = cp.Variable(nonneg=True)  # nonneg=True
        self.Vpar_min, self.Vpar_max = Vpar

        self._setup_prob()
        self.rng = np.random.default_rng(seed)

    def _init_X(self, X_init: np.ndarray) -> None:
        """If X_init is not already in 𝒳, then set self.var_X.value to Steiner
        point.
        """
        self.var_X.value = X_init  # this automatically checks that X_init is PSD
        total_violation = sum(np.sum(constraint.violation()) for constraint in self.X_set)
        if total_violation == 0:
            self.log.write(f'X_init valid.')
        else:
            self.log.write(f'X_init invalid. Violation: {total_violation:.3f}. Setting X_init = Steiner(𝒳).')

            n = self.n
            X = self.var_X

            theta = cp.Parameter(self.dim)  # vector
            obj = cp.Maximize(theta[:-n] @ cp.upper_tri(X) + theta[-n:] @ cp.diag(X))
            prob = cp.Problem(objective=obj, constraints=self.X_set)

            X_values = []
            for _ in range(self.nsamples_steiner):
                theta.value = self.rng.normal(size=self.dim)
                solve_prob(prob, log=self.log, name='CBC init')
                X_values.append(X.value.copy())
            X.value = np.mean(X_values, axis=0)

            make_pd_and_pos(self.var_X.value)
            total_violation = sum(np.sum(constraint.violation()) for constraint in self.X_set)
            self.log.write(f'After Steiner: X_init violation: {total_violation:.3f}.')

    def _setup_prob(self) -> None:
        """Defines self.prob as calculating the Steiner point of the consistent
        set.
        """
        n = self.n
        ub = self.eta  # * np.ones([n, 1])
        lb = -ub

        # optimization variable
        X = self.var_X

        constrs = self.X_set[:]  # make a shallow copy
        obs = self.obs_nodes
        self.param = {}

        theta = cp.Parameter(self.dim)  # vector
        self.param['theta'] = theta

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

        obj = cp.Maximize(theta[:-n] @ cp.upper_tri(X) + theta[-n:] @ cp.diag(X))
        self.prob = cp.Problem(objective=obj, constraints=constrs)

        # if cp.Problem is DPP, then it can be compiled for speedup
        # - http://cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming  # noqa
        self.log.write(f'CBC prob is DPP?: {self.prob.is_dcp(dpp=True)}')
        assert self.prob.is_dcp(dpp=True)

    def add_obs(self, t: int) -> None:
        """
        Args
        - t: int, current time step (>=1), v[t] and q[t] have just been updated

        Args
        - v: np.array, v(t+1) = v(t) + X @ u(t) = X @ q^c(t+1) + vpar(t+1)
        - u: np.array, u(t) = q^c(t+1) - q^c(t)
        """
        # update self.u and self.Δv
        super().add_obs(t)

        if self.is_cached:
            satisfied, msg = self._check_newest_obs(t)
            if not satisfied:
                self.is_cached = False
                self.log.write(f't = {t:6d}, CBC pre opt: {msg}')

        if self.prune_constraints:
            if t >= 2:
                check_informative(t=t-1, b=self.Δv, c=self.u, useful=self.w_inds)
                check_informative(t=t, b=self.v, c=self.q, useful=self.vpar_inds)

            # cmp_delta = (Δv <= self.Δv[self.w_inds_ub])
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
                self.log.write(f'active constraints - w: {num_w_inds}/{t}, '
                               f'vpar: {num_vpar_inds}/{t}')

    def _check_newest_obs(self, t: int) -> tuple[bool, str]:
        """Checks whether self.X_cache satisfies the newest observation:
        (v[t], q[t], u[t-1], Δv[t-1])

        Returns
        - satisfied: bool, whether self.X_cache satisfies the newest observation
        - msg: str, (if not satisfied) describes which constraints are violated
            (if satisfied) is empty string ''
        """
        obs = self.obs_nodes
        ŵ = self.Δv[t-1] - self.u[t-1] @ self.X_cache
        vpar_hat = self.v[t] - self.q[t] @ self.X_cache
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
        """
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
        """
        # be lazy if self.X_cache already satisfies the newest obs.
        if self.is_cached:
            return self.X_cache

        indent = ' ' * 11

        # optimization variables
        X = self.var_X
        # slack_w = self.var_slack_w

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

        prob = self.prob
        X_values = []
        for i in tqdm(range(self.nsamples_steiner)):
            self.param['theta'].value = self.rng.normal(size=self.dim)
            solve_prob(prob, log=self.log, name='CBC', indent=indent)
            X_values.append(X.value.copy())
        X.value = np.mean(X_values, axis=0)

        self.X_cache = np.array(X.value)  # make a copy
        make_pd_and_pos(self.X_cache)
        self.is_cached = True

        # check slack variable
        # if slack_w.value > 0:
        #     self.log.write(f'{indent} CBC slack: {slack_w.value:.3f}')

        # check whether constraints are satisfied for latest time step
        satisfied, msg = self._check_newest_obs(t)
        if not satisfied:
            self.log.write(f'{indent} CBC post opt: {msg}')

        return np.array(self.X_cache)  # return a copy


# def psd_steiner_point(num_samples: int, X: cp.Variable,
#                       constraints: list[Constraint]) -> np.ndarray:
#     """
#     Args
#     - num_samples: int, number of samples to use for calculating Steiner point
#     - X: cp.Variable, shape [n, n]
#     - constraints: list, cvxpy constraints on X
#     """
#     n = X.shape[0]
#     S = 0

#     param_theta = cp.Parameter(X.shape)
#     objective = cp.Maximize(cp.trace(param_theta @ X))
#     prob = cp.Problem(objective=objective, constraints=constraints)
#     assert prob.is_dcp(dpp=True)

#     rng = np.random.default_rng()

#     for i in range(num_samples):
#         theta = rng.random(X.shape)
#         theta = theta @ theta.T + 1e-7 * np.eye(n)  # random strictly PD matrix
#         theta /= np.linalg.norm(theta, 'fro')  # unit norm

#         param_theta.value = theta
#         prob.solve()
#         assert prob.status == 'optimal'

#         p_i = prob.value
#         S += p_i * theta

#     d = n + n*(n-1) // 2
#     S = S / num_samples * d

#     # check to make sure there is no constraint violation
#     X.value = S
#     for constr in constraints:
#         constr.violation()
#     raise NotImplementedError
#     return None
