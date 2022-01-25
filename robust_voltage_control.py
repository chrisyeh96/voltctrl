from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import IPython.display


class VoltPlot:

    TIME_TICKS =  [      0,    2400,    4800,    7200,    9600,   12000,   14400]  # noqa
    TIME_LABELS = ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00']  # noqa

    def __init__(self, v_lims: tuple[float, float],
                 q_lims: tuple[float, float], widget: bool | None = None):
        """
        Args
        - v_lims: tuple of float (v_min, v_max), units kV (not squared)
        - q_lims: tuple of float (q_min, q_max), units MVar
        - widget: optional bool, whether using `%matplotlib widget`
        """
        # Recreate Fig8 in Qu and Li (2020)
        # - they count the substation as bus 1
        # - we count the substation as bus -1
        self.index = [9, 19, 22, 31, 40, 46, 55]

        if widget is None:
            widget = ('nbagg' in plt.get_backend())
        self.widget = widget
        tqdm.write(f'widget? {self.widget}')

        fig, axs = plt.subplots(1, 4, figsize=(16, 4), tight_layout=True)
        self.fig = fig
        self.axs = axs

        q_min, q_max = q_lims
        v_min, v_max = v_lims

        ax = axs[0]
        ax.axhline(q_min, ls='--')
        ax.axhline(q_max, ls='--')
        ax.set_ylabel('Reactive Power (MVar)')
        ax.set_title('Reactive power injection')

        ax = axs[1]
        ax.axhline(v_min, ls='--')
        ax.axhline(v_max, ls='--')
        ax.set(ylabel='Voltage (kV)')
        ax.set_title('Voltage Profile')

        ax = axs[2]
        ax.axhline(v_min, ls='--')
        ax.axhline(v_max, ls='--')
        ax.set_ylabel('Voltage (kV)')
        ax.set_title('Voltage Profile without Controller')

        ax = axs[3]
        ax.set_ylabel(r'$||\hat{X} - X||_{\Delta}$')
        ax.set_title('Convergence of Consistent Model Chasing')

        for ax in axs:
            ax.set_xticks(VoltPlot.TIME_TICKS)
            ax.set_xticklabels(VoltPlot.TIME_LABELS)

        # create empty plots, placeholders
        self.qcs_lines = []
        self.vs_lines = []
        self.vs_noaction_lines = []
        self.dist_line = axs[3].step([], [], where='post')[0]  # step-function
        for i in np.asarray(self.index) - 2:
            qcs_line, = axs[0].plot([], [], label=f'bus {i+2}')
            vs_line, = axs[1].plot([], [])
            vs_noaction_line, = axs[2].plot([], [])

            self.qcs_lines.append(qcs_line)
            self.vs_lines.append(vs_line)
            self.vs_noaction_lines.append(vs_noaction_line)

        axs[0].legend()
        if widget:
            plt.show()
        else:
            plt.close()  # so that the plot doesn't show

    def update(self, qcs: np.ndarray, vs: np.ndarray, vs_noaction: np.ndarray,
               dists: tuple[list, list]) -> None:
        """
        Args
        - qcs: np.array, shape [n, T]
        - vs: np.array, shape [n, T]
        - vs_noaction: np.array, shape [n, T]
        """
        ts = range(qcs.shape[1])
        for l, i in enumerate(np.asarray(self.index) - 2):
            self.qcs_lines[l].set_data(ts, qcs[i])
            self.vs_lines[l].set_data(ts, vs[i])
            self.vs_noaction_lines[l].set_data(ts, vs_noaction[i])
        self.dist_line.set_data(dists)
        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()

    def show(self, clear_display: bool = False):
        if clear_display:
            IPython.display.clear_output()
        if self.widget:
            self.fig.canvas.draw()
        else:
            IPython.display.display(self.fig)


def robust_voltage_control(
        p: np.ndarray, qe: np.ndarray,
        v_lims: tuple[Any, Any], q_lims: tuple[Any, Any], v_nom: Any,
        X: np.ndarray, R: np.ndarray,
        Pv: np.ndarray, Pu: np.ndarray,
        eta: float, eps: float, v_sub: float, beta: float,
        sel: Any, volt_plot: VoltPlot | None = None) -> np.ndarray:
    """Runs robust voltage control.

    Args
    - p: np.array, shape [n, T], active power injection (MW)
    - qe: np.array, shape [n, T], exogenous reactive power injection (MVar)
    - v_lims: tuple (v_min, v_max), squared voltage magnitude limits (kV^2)
        - v_min, v_max could be floats, or np.arrays of shape [n]
    - q_lims: tuple (q_min, q_max), reactive power injection limits (MVar)
        - q_min, q_max could be floats, or np.arrays of shape [n]
    - v_nom: float or np.array of shape [n], desired nominal voltage
    - X: np.array, shape [n, n], line parameters for reactive power injection
    - R: np.array, shape [n, n], line parameters for active power injection
    - Pv: np.array, shape [n, n], quadratic (PSD) cost matrix for voltage
    - Pu: np.array, shape [n, n], quadratic (PSD) cost matrix for control
    - eta: float, noise bound (kV^2)
    - eps: float, robustness buffer (kV^2)
    - v_sub: float, fixed squared voltage magnitude at substation (kV^2)
    - sel: nested convex body chasing object (e.g., CBCProjection)
    - volt_plot: VoltPlot

    Returns: TODO
    """
    assert p.shape == qe.shape
    n, T = qe.shape

    dists = {'t': [], 'true': [], 'prev': []}
    Xhat_prev = None

    v_min, v_max = v_lims
    q_min, q_max = q_lims
    if isinstance(v_min, float):
        v_min = np.ones(n) * v_min
        v_max = np.ones(n) * v_max

    if isinstance(q_min, float):
        rho = eps / (2 * (q_max - q_min) * np.sqrt(n))
    else:
        rho = eps / (2 * np.linalg.norm(q_max - q_min, ord=2))
    print(f'rho(eps={eps:.2f}) = {rho:.3f}')

    vs = np.zeros_like(p)  # vs[:, t] denotes v(t)
    qcs = np.zeros_like(p)  # qcs[:, t] denotes q^c(t)
    vs_noaction = X @ qe + R @ p + v_sub
    vs[:, 0] = vs_noaction[:, 0]

    # we need to use `u` as the variable instead of `qc_next` in order to
    # make the problem DPP-convex
    u = cp.Variable(n)
    slack = cp.Variable(nonneg=True)

    vt = cp.Parameter(n)
    qct = cp.Parameter(n)
    Xhat = cp.Parameter([n, n], nonneg=True)

    qc_next = qct + u
    v_next = vt + Xhat @ u
    pad = eta + rho * cp.norm(u, p=1)

    obj = cp.Minimize(cp.quad_form(v_next - v_nom, Pv)
                      + cp.quad_form(u, Pu)
                      + beta * slack**2)
    constraints = [
        q_min <= qc_next, qc_next <= q_max,
        v_min + pad - slack <= v_next, v_next <= v_max - pad + slack
    ]
    prob = cp.Problem(objective=obj, constraints=constraints)

    # if CBC problem is DPP, then it can be compiled for speedup
    # - see https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming  # noqa
    tqdm.write(f'CBC prob is DPP?: {prob.is_dcp(dpp=True)}')

    for t in tqdm(range(T-1)):
        # fill in Parameters
        Xhat.value = sel.select()
        qct.value = qcs[:, t]
        vt.value = vs[:, t]

        update_dists(dists, t, Xhat.value, Xhat_prev, X)
        Xhat_prev = Xhat.value

        try:
            prob.solve(warm_start=True)
        except cp.SolverError:
            tqdm.write('robust oracle: default solver failed. Trying cp.ECOS')
            prob.solve(solver=cp.ECOS)
        if prob.status != 'optimal':
            tqdm.write(f'robust oracle: prob.status = {prob.status}')
            if 'infeasible' in prob.status:
                import pdb
                pdb.set_trace()

        qcs[:, t+1] = qc_next.value
        vs[:, t+1] = vs_noaction[:, t+1] + X @ qc_next.value
        sel.add_obs(v=vs[:, t+1], u=u.value)
        # tqdm.write(f't = {t}, ||u||_1 = {np.linalg.norm(u.value, 1)}')

        if volt_plot is not None and (t+1) % 500 == 0:
            volt_plot.update(qcs=qcs[:, :t+2],
                             vs=np.sqrt(vs[:, :t+2]),
                             vs_noaction=np.sqrt(vs_noaction[:, :t+2]),
                             dists=(dists['t'], dists['true']))
            volt_plot.show()
    return vs, qcs


def np_triangle_norm(x: np.ndarray) -> float:
    """Computes ||X||_△"""
    return np.linalg.norm(np.triu(x), ord='fro')


def update_dists(dists: dict[str, list], t: int, Xhat: np.ndarray,
                 Xhat_prev: np.ndarray | None, X: np.ndarray) -> None:
    """Calculates ||Xhat-X||_△ and ||Xhat-Xhat_prev||_△.

    Args
    - dists: dict, keys ['t', 'true', 'prev'], values are lists
    - t: int, time step
    - Xhat: np.array, shape [n, n]
    - Xhat_prev: np.array, shape [n, n]
    - X: np.array, shape [n, n]
    """
    # here, we rely on the fact that the CBCProjection returns the existing
    # variable Xhat if it doesn't need to move
    if Xhat is not Xhat_prev:
        dist_true = np_triangle_norm(Xhat - X)
        if Xhat_prev is None:
            dist_prev = 0
        else:
            dist_prev = np_triangle_norm(Xhat - Xhat_prev)
        tqdm.write(f't = {t:6d}, ||Xhat-X||_△ = {dist_true:7.1f}, '
                   f'||Xhat-Xhat_prev||_△ = {dist_prev:5.3f}')

        dists['t'].append(t)
        dists['true'].append(dist_true)
        dists['prev'].append(dist_prev)
