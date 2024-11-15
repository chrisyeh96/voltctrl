from __future__ import annotations

from collections.abc import Sequence
import io
from typing import Any

import cvxpy as cp
import numpy as np
from tqdm.auto import tqdm

from cbc.base import CBCInfeasibleError
from network_utils import np_triangle_norm
from utils import solve_prob
from voltplot import VoltPlot


def robust_voltage_control(
        vpars: np.ndarray,
        v_lims: tuple[Any, Any], q_lims: tuple[Any, Any], v_nom: Any,
        X: np.ndarray, require_X_psd: bool,
        Pv: np.ndarray, Pu: np.ndarray,
        eta: float, ε: float, β: float,
        sel: Any, δ: float = 0.,
        ctrl_nodes: Sequence[int] | None = None,
        pbar: tqdm | None = None,
        log: tqdm | io.TextIOBase | None = None,
        volt_plot: VoltPlot | None = None, volt_plot_update: int = 100,
        save_params_every: int = 100,
        change_net: tuple[int, np.ndarray] | None = None
        ) -> tuple[np.ndarray, np.ndarray, dict[str, list],
                   dict[int, np.ndarray | tuple[np.ndarray, float]],
                   tuple[list, list]]:
    """Runs robust voltage control.

    If change_net is not None, then the order of events is:
        sel.select(change_t-1)
        qcs[change_t] is decided by robust controller
        <TOPOLOGY CHANGE>
        vs[change_t] = vpars_mod[change_t] + qcs[change_t] @ X_mod
        sel.add_obs(change_t)
        sel.select(change_t)  # this should raise CBCInfeasibleError

    Args
    - vpars: np.array, shape [T, n], uncontrolled squared voltages (kV^2)
    - v_lims: tuple (v_min, v_max), squared voltage magnitude limits (kV^2)
        - v_min, v_max could be floats, or np.arrays of shape [n]
    - q_lims: tuple (q_min, q_max), reactive power injection limits (MVar)
        - q_min, q_max could be floats, or np.arrays of shape [n]
    - v_nom: float or np.array of shape [n], desired nominal voltage
    - X: np.array, shape [n, n], true line parameters for reactive power injection
    - require_X_psd: bool, whether to require that selected X is always PSD
        - should usually be True, unless using least-squares controller
    - Pv: np.array, shape [n, n], quadratic (PSD) cost matrix for voltage
    - Pu: np.array, shape [n, n], quadratic (PSD) cost matrix for control
    - eta: float, noise bound (kV^2)
    - ε: float, robustness buffer (kV^2)
    - β: float, weight for slack variable
    - sel: nested convex body chasing object (e.g., CBCProjection)
    - δ: float, weight of noise term in CBC norm when learning eta
        - set to 0 if eta is known
    - ctrl_nodes: list of int, nodes that we can control voltages for
    - pbar: optional tqdm, progress bar
    - log: optional log output
    - volt_plot: VoltPlot
    - volt_plot_update: int, time steps between updating volt_plot
    - save_params_every: int, time steps between saving estimated model params
    - change_net: tuple (change_t, X_mod)
        - change_t: time step when X changes to X_mod
        - X_mod: new X matrix

    Returns
    - vs: np.array, shape [T, n]
    - qcs: np.array, shape [T, n]
    - dists: dict, keys ['t', '*_true', '*_prev'], values are lists
        - 't': list of int, time steps at which model updates occurred,
            i.e., X̂(t) != X̂(t-1). X̂(t) is generated by data up to and
            including v(t), q^c(t), u(t-1)
        - '*_true': list of float, ‖X̂-X‖_△ after each model update
            (and likewise for η and (X,η), if learning η)
        - '*_prev': list of float, ‖X̂(t)-X̂(t-1)‖_△ after each model update
            (and likewise for η and (X,η), if learning η)
    - params: dict, keys are time step t, values the estimated model params
        after observing vs[t], qcs[t].
        - if delta is None: np.array, shape [n, n]
        - if delta is given: tuple of (np.ndarray, float)
    - consistent_arrs: tuple (consistent, consistent_05)
        - consistent: whether parameters are consistent
        - consistent_05: allows for 0.05 in vpar constraint violation, because
            CVXPY empirically may still have up to 0.05 of constraint violation,
            even when it solves SEL to "optimality"
    """
    T, n = vpars.shape

    if log is None:
        log = tqdm()

    log.write(f'‖X‖_△ = {np_triangle_norm(X):.2f}')

    dists: dict[str, list] = {'t': [], 'X_true': [], 'X_prev': []}
    X̂_prev = None

    v_min, v_max = v_lims
    q_min, q_max = q_lims

    vs = sel.v  # shape [T, n], vs[t] denotes v(t)
    qcs = sel.q  # shape [T, n], qcs[t] denotes q^c(t)

    if not np.array_equal(vs[0], vpars[0]):
        # usually, we want vs[0] == vpars[0]
        # but this might not be true when the network changes
        log.write('Not np.array_equal(vs[0], vpars[0]) - is there a bug?')

    # we need to use `u` as the variable instead of `qc_next` in order to
    # make the problem DPP-convex
    u = cp.Variable(n, name='u')
    ξ = cp.Variable(nonneg=True, name='ξ')  # slack variable

    q_norm_2 = np.linalg.norm(np.ones(n) * (q_max-q_min), ord=2)
    if δ > 0:  # learning eta
        dists |= {'η': [], 'η_prev': [], 'X_η_prev': []}
        etahat_prev = None
        rho = ε * δ / (1 + δ * q_norm_2)
        etahat = cp.Parameter(nonneg=True, name='̂η')
        k = etahat + rho * (1/δ + cp.norm2(u))
    else:
        rho = ε / q_norm_2
        k = eta + rho * cp.norm(u, p=2)
    log.write(f'rho(ε={ε:.2f}) = {rho:.3f}')

    # parameters are placeholders for given values
    vt = cp.Parameter(n, name='v')
    qct = cp.Parameter(n, name='qc')
    X̂ = cp.Parameter((n, n), PSD=require_X_psd, name='X̂')

    qc_next = qct + u
    v_next = vt + u @ X̂

    obj = cp.Minimize(cp.quad_form(v_next - v_nom, Pv)
                      + cp.quad_form(u, Pu)
                      + β * ξ**2)
    constraints = [
        q_min <= qc_next, qc_next <= q_max,
        v_min + k - ξ <= v_next, v_next <= v_max - k + ξ
    ]
    if ctrl_nodes is not None:
        all_nodes = np.arange(n)
        unctrl_nodes = np.setdiff1d(all_nodes, ctrl_nodes).tolist()
        constraints.append(u[unctrl_nodes] == 0)
    prob = cp.Problem(objective=obj, constraints=constraints)

    # if cp.Problem is DPP, then it can be compiled for speedup
    # - http://cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming
    assert prob.is_dcp(dpp=True)
    log.write(f'Robust Oracle prob is DPP?: {prob.is_dcp(dpp=True)}')

    if pbar is not None:
        log.write('pbar present')
        pbar.reset(total=T-1)

    params: dict[int, np.ndarray | tuple[np.ndarray, float]] = {}
    consistent_arr = []
    consistent_05_arr = []
    for t in range(T-1):  # t = 0, ..., T-2
        # fill in Parameters
        if δ > 0:  # learning eta
            try:
                X̂.value, etahat.value = sel.select(t)
                consistent, consistent_05 = sel.is_consistent(t, X, eta)
            except CBCInfeasibleError:
                break
            update_dists(dists, t, X_info=(X̂.value, X̂_prev, X),
                         η_info=(etahat.value, etahat_prev, eta), δ=δ, log=log)
            etahat_prev = float(etahat.value)  # save a copy
            if (t+1) % save_params_every == 0:
                params[t] = (np.array(X̂.value), etahat_prev)
        else:
            try:
                X̂.value = sel.select(t)
                consistent, consistent_05 = sel.is_consistent(t, X)
            except CBCInfeasibleError:
                break
            update_dists(dists, t, X_info=(X̂.value, X̂_prev, X), log=log)
            if (t+1) % save_params_every == 0:
                params[t] = np.array(X̂.value)  # save a copy

        consistent_arr.append(consistent)
        consistent_05_arr.append(consistent_05)

        X̂_prev = np.array(X̂.value)  # save a copy
        qct.value = qcs[t]
        vt.value = vs[t]

        solve_prob(prob, log=log, name=f't={t}. robust oracle')
        if prob.status == 'infeasible':
            raise RuntimeError('robust controller infeasible')

        qcs[t+1] = qc_next.value

        if change_net is None or t+1 < change_net[0]:
            vs[t+1] = vpars[t+1] + qc_next.value @ X
        elif t+1 >= change_net[0]:
            X_mod = change_net[1]
            vs[t+1] = vpars[t+1] + qc_next.value @ X_mod
        sel.add_obs(t+1)
        # log.write(f't = {t}, ‖u‖_1 = {np.linalg.norm(u.value, 1)}')

        if volt_plot is not None and (t+1) % volt_plot_update == 0:
            volt_plot.update(qcs=qcs[:t+2],
                             vs=np.sqrt(vs[:t+2]),
                             vpars=np.sqrt(vpars[:t+2]),
                             dists=(dists['t'], dists['X_true']))
            volt_plot.show(clear_display=False)

        if pbar is not None:
            pbar.update()
        if (t+1) % volt_plot_update == 0:
            log.write(f't={t}. robust oracle progress.')

    # update voltplot at the end of run
    if volt_plot is not None:
        volt_plot.update(qcs=qcs, vs=np.sqrt(vs), vpars=np.sqrt(vpars),
                         dists=(dists['t'], dists['X_true']))
        volt_plot.show(clear_display=False)

    return vs, qcs, dists, params, (consistent_arr, consistent_05_arr)


def np_triangle_delta_norm(X: np.ndarray, η: float, δ: float) -> float:
    X_norm = np_triangle_norm(X)
    return np.sqrt(X_norm**2 + (δ * η)**2)


def update_dists(dists: dict[str, list], t: int,
                 X_info: tuple[np.ndarray, np.ndarray | None, np.ndarray],
                 η_info: tuple[float, float | None, float] | None = None,
                 δ: float = 0., log: tqdm | io.TextIOBase | None = None,
                 ) -> None:
    """Calculates ‖X̂-X‖_△ and ‖X̂-X̂_prev‖_△.

    Args
    - dists: dict, keys ['t', '*_true', '*_prev'], values are lists
        - 't': list of int, time steps at which model updates occurred,
            i.e., X̂(t) != X̂(t-1). X̂(t) is generated by data up to and
            including v(t), q^c(t), u(t-1)
        - '*_true': list of float, ‖X̂-X‖_△ after each model update
            (and likewise for η and (X,η), if learning η)
        - '*_prev': list of float, ‖X̂(t)-X̂(t-1)‖_△ after each model update
            (and likewise for η and (X,η), if learning η)
    - t: int, time step
    - X_info: tuple of (X̂, X̂_prev, X*), each is np.array of shape [n,n]
        - X̂_prev may be None on the 1st time step
    - η_info: tuple of (̂η, ̂η_prev, ηmax), each is float
        - ̂η_prev may be None on the 1st time step
    - δ: float, weight of noise term in CBC norm when learning eta
    - log: optional log file
    """
    X̂, X̂_prev, X = X_info
    if δ > 0:
        assert η_info is not None
        etahat, etahat_prev, η = η_info

    # here, we rely on the fact that the CBCProjection returns the existing
    # parameter if it doesn't need to move
    if X̂_prev is not None and np.array_equal(X̂, X̂_prev):
        if δ == 0. or (etahat_prev is not None and etahat == etahat_prev):
            return

    dists['t'].append(t)
    msg = f't = {t:6d}'

    if δ > 0:
        # dXη = np_triangle_delta_norm(X̂ - X, etahat - η, δ)
        # msg += f', ‖(X̂,̂η)-(X,η)‖_(△,δ) = {dXη:7.3f}'
        # dists['X_η_true'].append(dXη)

        # dη = np.abs(etahat - η)
        # msg += f', |̂η-η| = {dη:3.3f}'
        # dists['η_true'].append(dη)

        if X̂_prev is None:
            dXη = 0.
            dη = 0.
        else:
            assert etahat_prev is not None
            dXη = np_triangle_delta_norm(X̂ - X̂_prev, etahat - etahat_prev, δ)
            dη = np.abs(etahat - etahat_prev)
            msg += f', ‖(X̂,̂η)-(X̂,̂η)_prev‖_(△,δ) = {dXη:5.3f}'
            msg += f', |̂η-̂η_prev| = {dη:3.3f}'
        dists['X_η_prev'].append(dXη)
        dists['η_prev'].append(dη)
        dists['η'].append(etahat)

    dX = np_triangle_norm(X̂ - X)
    msg += f', ‖X̂-X‖_△ = {dX:7.3f}'
    dists['X_true'].append(dX)

    if X̂_prev is None:
        dX = 0.
    else:
        dX = np_triangle_norm(X̂ - X̂_prev)
        msg += f', ‖X̂-X̂_prev‖_△ = {dX:5.3f}'
    dists['X_prev'].append(dX)

    if log is None:
        log = tqdm
    log.write(msg)
