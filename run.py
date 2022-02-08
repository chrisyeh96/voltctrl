from __future__ import annotations

from collections.abc import Callable
import pickle
import time
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
        is_interactive: bool = False):
    """
    Args
    - eta: float, maximum ||w||∞
    - epsilon: float, robustness
    - q_max: float, maximum reactive power injection
    - norm_bound: float
    - noise: float, network impedances modified by fraction Uniform(±noise)
    - seed: int, random seed
    """
    params: dict[str, Any] = {
        'qmax': q_max,
        'eta': eta,
        'eps': epsilon,
        'norm': norm_bound,
        'CBC': cbc_alg,
        'seed': seed
    }

    # read in data
    if noise > 0:
        params['noise'] = noise
    net = create_56bus()
    R, X = create_RX_from_net(net, noise=noise, seed=seed)
    p, qe = read_load_data()  # in MW and MVar
    n, T = p.shape

    ### FIXED PARAMETERS
    v_min, v_max = (11.4**2, 12.6**2)  # +/-5%, units kV^2
    v_nom = 12**2  # nominal squared voltage magnitude, units kV^2
    v_sub = v_nom  # fixed squared voltage magnitude at substation, units kV^2

    vpars = X @ qe + R @ p + v_sub  # shape [n, T]
    Vpar_min = np.min(vpars, axis=1)  # shape [n]
    Vpar_max = np.max(vpars, axis=1)  # shape [n]

    Pv = 0.1 * np.eye(n)
    Pu = 10 * np.eye(n)

    # weights on slack variables: alpha for CBC, beta for robust oracle
    alpha = 1000
    beta = 10
    ### end of FIXED PARAMETERS

    start = 0

    # randomly initialize a PSD and entry-wise positive matrix
    # with the same norm as the true X
    rng = np.random.default_rng(seed=seed)
    X_init = rng.normal(size=(n, n))
    make_pd_and_pos(X_init)
    X_init *= np_triangle_norm(X) / np_triangle_norm(X_init)

    save_dict = {
        'X_init': X_init
    }

    if cbc_alg == 'const':
        sel = CBCBase(X_init)
    elif cbc_alg == 'proj':
        params['nsamples'] = nsamples
        gen_X_set = meta_gen_X_set(norm_bound=norm_bound, X_true=X)
        sel = CBCProjection(
            eta=eta, n=n, T=T-start, nsamples=nsamples, alpha=alpha,
            v=vpars[:, start], gen_X_set=gen_X_set, Vpar=(Vpar_min, Vpar_max),
            X_init=X_init, X_true=X)
    else:
        raise ValueError('unknown cbc_alg')

    volt_plot = VoltPlot(v_lims=(np.sqrt(v_min), np.sqrt(v_max)), q_lims=(-q_max, q_max))

    vs, qcs, dists = robust_voltage_control(
        p=p[:, start:T], qe=qe[:, start:T],
        v_lims=(v_min, v_max), q_lims=(-q_max, q_max), v_nom=v_nom,
        X=X, R=R, Pv=Pv, Pu=Pu, eta=eta, eps=epsilon, v_sub=v_sub, beta=beta,
        sel=sel,
        volt_plot=volt_plot if is_interactive else None)

    if isinstance(sel, CBCProjection):
        save_dict.update({
            'w_inds': sel.w_inds,
            'vpar_inds': sel.vpar_inds
        })

    # TODO: check whether this is actually necessary
    dists['t'].append(T-1)
    dists['true'].append(dists['true'][-1])

    volt_plot.update(qcs=qcs,
                     vs=np.sqrt(vs),
                     vpars=np.sqrt(vpars),
                     dists=(dists['t'], dists['true']))

    # save figure
    filename = '_'.join(f'{k}{v}' for k,v in params.items())
    volt_plot.fig.savefig(f'{filename}.svg', pad_inches=0, bbox_inches='tight')
    volt_plot.fig.savefig(f'{filename}.pdf', pad_inches=0, bbox_inches='tight')

    # save data
    np.savez_compressed(
        f'{filename}.npz',
        vs=vs, qcs=qcs, dists=dists, **save_dict)


if __name__ == '__main__':
    run(
        epsilon=0.1,
        q_max=0.2,
        cbc_alg='const',
        eta=8.65,
        norm_bound=0.2,
        noise=0,
        seed=123,
        is_interactive=False)