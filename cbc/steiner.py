"""Convex body chasing via Steiner point."""
from __future__ import annotations

from collections.abc import Callable
import io

import cvxpy as cp
import numpy as np
from tqdm.auto import tqdm

from cbc.base import CBCBase
from network_utils import make_pd_and_pos

Constraint = cp.constraints.constraint.Constraint


class CBCSteiner(CBCBase):
    """Finds the set of X that is consistent with the observed data. Assumes
    that noise bound (eta) is known.
    """
    def __init__(self, n: int, T: int, X_init: np.ndarray, v: np.ndarray,
                 gen_X_set: Callable[[cp.Variable], list[Constraint]],
                 eta: float, nsamples: int, nsamples_steiner: int, # alpha: float,
                 Vpar: tuple[np.ndarray, np.ndarray],
                 X_true: np.ndarray,
                 log: tqdm | io.TextIOBase | None = None, seed: int = 123):
        """
        Args
        - see CBCBase for descriptions of other parameters
        - eta: float, noise bound
        - nsamples: int, # of observations to use for defining the convex set
        - nsamples_steiner: int, # of random directions to use for estimating the
            Steiner point integral
        # - alpha: float, weight on slack variable
        - Vpar: tuple (Vpar_min, Vpar_max), box description of Vpar
            - each Vpar_* is a np.array of shape [n]
        - seed: int, random seed
        """
        super().__init__(n=n, T=T, X_init=X_init, v=v, gen_X_set=gen_X_set,
                         X_true=X_true, log=log)
        self.dim = n * (n+1) // 2
        self.is_cached = True

        self.eta = eta
        self.nsamples = nsamples
        self.nsamples_steiner = nsamples_steiner
        # self.alpha = alpha

        self.w_inds = np.zeros([2, T-1], dtype=bool)  # whether each (u(t), delta_v(t)) is useful
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

            theta = cp.Parameter((self.dim,))  # vector
            obj = cp.Maximize(theta[:-n] @ cp.upper_tri(X) + theta[-n:] @ cp.diag(X))
            prob = cp.Problem(objective=obj, constraints=self.X_set)

            X_values = []
            for i in range(self.nsamples_steiner):
                theta.value = self.rng.normal(size=self.dim)
                prob.solve(solver=cp.MOSEK)
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

        constrs = self.X_set
        self.param = {}
        for b in ['lb', 'ub']:
            vs = cp.Parameter((self.nsamples, n), name=f'vs_{b}')
            delta_vs = cp.Parameter((self.nsamples, n), name=f'delta_vs_{b}')
            us = cp.Parameter((self.nsamples, n), name=f'us_{b}')
            qs = cp.Parameter((self.nsamples, n), name=f'qs_{b}')

            w_hats = delta_vs - us @ X
            vpar_hats = vs - qs @ X

            if b == 'lb':
                constrs.extend([lb <= w_hats,
                                self.Vpar_min[None, :] <= vpar_hats])
            else:
                constrs.extend([w_hats <= ub,
                                vpar_hats <= self.Vpar_max[None, :]])

            self.param[f'vs_{b}'] = vs
            self.param[f'delta_vs_{b}'] = delta_vs
            self.param[f'us_{b}'] = us
            self.param[f'qs_{b}'] = qs

        theta = cp.Parameter((self.dim,))  # vector
        obj = cp.Maximize(theta[:-n] @ cp.upper_tri(X) + theta[-n:] @ cp.diag(X))
        self.prob = cp.Problem(objective=obj, constraints=constrs)

        self.param['theta'] = theta

        # if cp.Problem is DPP, then it can be compiled for speedup
        # - http://cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming  # noqa
        self.log.write(f'CBC prob is DPP?: {self.prob.is_dcp(dpp=True)}')
        assert self.prob.is_dcp(dpp=True)

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

        # optimization variables
        X = self.var_X
        # slack_w = self.var_slack_w

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
                w_inds = self.w_inds[i].nonzero()[0]
                ts = np.concatenate([
                    w_inds[-k:],
                    self.rng.choice(len(w_inds) - k, size=self.nsamples-k, replace=False)
                ])
                self.param[f'delta_vs_{b}'].value = self.delta_v[ts]
                self.param[f'us_{b}'].value = self.u[ts]

                vpar_inds = self.vpar_inds[i].nonzero()[0]
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

        prob = self.prob
        X_values = []
        for i in tqdm(range(self.nsamples_steiner)):
            self.param['theta'].value = self.rng.normal(size=self.dim)
            prob.solve(solver=cp.MOSEK)
            X_values.append(X.value.copy())
        X.value = np.mean(X_values, axis=0)

        if prob.status != 'optimal':
            self.log.write(f'{indent} CBC prob.status = {prob.status}')
            if prob.status == 'infeasible':
                import pdb
                pdb.set_trace()
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
