from __future__ import annotations

from collections.abc import Callable
import pickle
import datetime as dt
from typing import Any

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from cbc import CBCBase, CBCProjection, cp_triangle_norm_sq
from network_utils import (
    create_56bus,
    create_RX_from_net,
    read_load_data,
    make_pd_and_pos
)
from robust_voltage_control import (
    VoltPlot,
    np_triangle_norm,
    robust_voltage_control
)

Constraint = cp.constraints.constraint.Constraint

# hide top and right splines on plots
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False


def meta_gen_X_set(norm_bound: float, X_true: np.ndarray
                   ) -> Callable[[cp.Variable], list[Constraint]]:
    """Creates a function that, given a cp.Variable respresenting X,
    returns constraints that describe its uncertainty set 𝒳.

    Args
    - norm_bound: float, parameter c such that
        ||var_X - X*||_△ <= c * ||X*||_△
    - X_true: np.ndarray, PSD matrix of shape [n, n]

    Returns: function
    """
    def gen_𝒳(var_X: cp.Variable) -> list[Constraint]:
        """Returns constraints describing 𝒳, the uncertainty set for X.

        Constraints:
        - var_X is PSD (enforced at cp.Variable intialization)
        - var_X is entry-wise nonnegative
        - ||var_X - X*|| <= c * ||X*||

        Args
        - var_X: cp.Variable, should already be constrainted to be PSD

        Returns: list of Constraint
        """
        assert var_X.is_psd(), 'variable for X was not PSD-constrained'
        norm_sq_diff = cp_triangle_norm_sq(var_X - X_true)
        norm_X = np_triangle_norm(X_true)
        𝒳 = [var_X >= 0,  # entry-wise nonneg
             norm_sq_diff <= (norm_bound * norm_X)**2]
        tqdm.write('𝒳 = {X: ||X̂-X||_△ <= ' + f'{norm_bound * norm_X}' + '}')
        return 𝒳
    return gen_𝒳


def run(epsilon: float, q_max: float, cbc_alg: str,
        eta: float, norm_bound: float,
        noise: float = 0, nsamples: int = 100, seed: int = 123,
        is_interactive: bool = False) -> None:
    """
    Args
    - eta: float, maximum ||w||∞
    - epsilon: float, robustness
    - q_max: float, maximum reactive power injection
    - norm_bound: float
    - noise: float, network impedances modified by fraction Uniform(±noise)
    - seed: int, random seed
    """
    start_time = dt.datetime.now()
    params: dict[str, Any] = dict(
        cbc_alg=cbc_alg, q_max=q_max, epsilon=epsilon, eta=eta)
    filename = f'CBC{cbc_alg}'

    # read in data
    if noise > 0:
        params.update(noise=noise, seed=seed, norm_bound=norm_bound)
        filename += f'_noise{noise}_norm{norm_bound}_seed{seed}'

    net = create_56bus()
    R, X = create_RX_from_net(net, noise=0)  # true R and X
    p, qe = read_load_data()  # in MW and MVar
    T, n = p.shape

    ### FIXED PARAMETERS
    v_min, v_max = (11.4**2, 12.6**2)  # +/-5%, units kV^2
    v_nom = 12**2  # nominal squared voltage magnitude, units kV^2
    v_sub = v_nom  # fixed squared voltage magnitude at substation, units kV^2

    vpars = qe @ X + p @ R + v_sub  # shape [T, n]
    Vpar_min = np.min(vpars, axis=1)  # shape [n]
    Vpar_max = np.max(vpars, axis=1)  # shape [n]

    Pv = 0.1
    Pu = 10

    # weights on slack variables: alpha for CBC, beta for robust oracle
    alpha = 1000
    beta = 100

    params.update(
        v_min=v_min, v_max=v_max, v_nom=v_nom, Pv=Pv, Pu=Pu, beta=beta)
    ### end of FIXED PARAMETERS

    start = 0

    # randomly initialize a network matrix
    _, X_init = create_RX_from_net(net, noise=noise, check_pd=True, seed=seed)

    save_dict = {
        'X_init': X_init
    }

    gen_X_set = meta_gen_X_set(norm_bound=norm_bound, X_true=X)

    if cbc_alg == 'const':
        sel = CBCBase(n=n, T=T, X_init=X_init, v=vpars[start],
                      gen_X_set=gen_X_set, X_true=X)
    elif cbc_alg == 'proj':
        params.update(alpha=alpha, nsamples=nsamples)
        sel = CBCProjection(
            eta=eta, n=n, T=T-start, nsamples=nsamples, alpha=alpha,
            v=vpars[start], gen_X_set=gen_X_set, Vpar=(Vpar_min, Vpar_max),
            X_init=X_init, X_true=X, seed=seed)
        save_dict.update(w_inds=sel.w_inds, vpar_inds=sel.vpar_inds)
    else:
        raise ValueError('unknown cbc_alg')

    volt_plot = VoltPlot(
        v_lims=(np.sqrt(v_min), np.sqrt(v_max)),
        q_lims=(-q_max, q_max))

    vs, qcs, dists = robust_voltage_control(
        p=p[start:T], qe=qe[start:T],
        v_lims=(v_min, v_max), q_lims=(-q_max, q_max), v_nom=v_nom,
        X=X, R=R,
        Pv=Pv * np.eye(n), Pu=Pu * np.eye(n),
        eta=eta, eps=epsilon, v_sub=v_sub, beta=beta, sel=sel,
        volt_plot=volt_plot if is_interactive else None)

    elapsed = (dt.datetime.now() - start_time).total_seconds()

    # save data
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=-8)))  # PST
    filename += now.strftime('_%Y%m%d_%H%M%S')
    with open(f'{filename}.pkl', 'wb') as f:
        pickle.dump(file=f, obj=dict(
            vs=vs, qcs=qcs, dists=dists, params=params,
            elapsed=elapsed, **save_dict))
    # np.savez_compressed(f'{filename}.npz', vs=vs, qcs=qcs, dists=dists, **save_dict)

    # plot and save figure
    volt_plot.update(qcs=qcs,
                     vs=np.sqrt(vs),
                     vpars=np.sqrt(vpars),
                     dists=(dists['t'], dists['true']))
    volt_plot.fig.savefig(f'{filename}.svg', pad_inches=0, bbox_inches='tight')
    volt_plot.fig.savefig(f'{filename}.pdf', pad_inches=0, bbox_inches='tight')


if __name__ == '__main__':
    run(
        epsilon=0.1,
        q_max=0.24,
        cbc_alg='const',
        eta=8.65,
        norm_bound=0.2,
        noise=0,
        seed=123,
        is_interactive=False)
