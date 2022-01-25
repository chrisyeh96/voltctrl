from __future__ import annotations

from typing import TypeVar

import networkx as nx
import numpy as np
import pandapower as pp
import pandapower.topology


T = TypeVar('T')


def create_56bus() -> pp.pandapowerNet:
    """
    Creates the SCE 56-bus network from the MATPOWER file.
    Bus 0 is the substation, and the other buses are numbered 1-55.
    At every bus (except 0), we attach a load and static generator element.

    Returns: pp.pandapowerNet
    """
    net = pp.converter.from_mpc('data/SCE_56bus.mat', casename_mpc_file='case_mpc')

    # remove loads and generators at all buses except bus 0 (substation),
    # but keep the network lines
    buses = list(range(1, 56))
    pp.drop_elements_at_buses(net, buses=buses, bus_elements=True, branch_elements=False)

    for i in buses:
        pp.create_load(net, bus=i, p_mw=0, q_mvar=0)
        pp.create_sgen(net, bus=i, p_mw=0, q_mvar=0)

    return net


def create_R_X_from_net(net: pp.pandapowerNet) -> tuple[np.ndarray, np.ndarray]:
    """Creates R,X matrices from a pandapowerNet.

    Args
    - net: pandapowerNet with (n+1) buses including substation

    Returns: tuple (X, R)
    - X: np.array, shape [n, n], positive definite and entry-wise positive
    - R: np.array, shape [n, n], positive definite and entry-wise positive
    """
    # read in r and x matrices from data
    n = len(net.bus) - 1  # number of buses, excluding substation
    r = np.ones((n+1, n+1)) * np.inf
    x = np.ones((n+1, n+1)) * np.inf

    r[net.line['from_bus'], net.line['to_bus']] = net.line['r_ohm_per_km']
    r[net.line['to_bus'], net.line['from_bus']] = net.line['r_ohm_per_km']
    x[net.line['from_bus'], net.line['to_bus']] = net.line['x_ohm_per_km']
    x[net.line['to_bus'], net.line['from_bus']] = net.line['x_ohm_per_km']

    G = pp.topology.create_nxgraph(net)
    R, X = create_R_X(r, x, G)
    return R, X


def get_intersecting_path(path1: Sequence[T], path2: Sequence[T]) -> list[tuple[T, T]]:
    """Gets the intersection between two paths. Assumes that the paths only
    intersect in the beginning.

    Args
    - path1: list of int
    - path2: list of int

    Returns: list of tuple, edges in the intersecting path
    """
    ret = []
    for k in range(1, min(len(path1), len(path2))):
        u = path1[k]
        v = path2[k]
        if u == v:
            edge = (path1[k-1], u)
            ret.append(edge)
        else:
            break
    return ret


def is_pos_def(A: np.ndarray) -> bool:
    """Checks whether a matrix is positive definite.

    Args
    - A: np.array, matrix

    Returns: bool, true iff A>0
    """
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def create_R_X(r: np.ndarray, x: np.ndarray, G: nx.Graph
              ) -> tuple[np.ndarray, np.ndarray]:
    """Creates R,X matrices from line impedance matrices r and x.

    Args
    - r: np.array, shape [n+1, n+1], symmetric and entry-wise positive
    - x: np.array, shape [n+1, n+1], symmetric and entry-wise positive
    - G: nx.Graph, graph

    Returns: tuple (X, R)
    - X: np.array, shape [n, n], positive definite and entry-wise positive
    - R: np.array, shape [n, n], positive definite and entry-wise positive
    """
    n = r.shape[0] - 1

    R = np.zeros((n+1, n+1), dtype=float)
    X = np.zeros((n+1, n+1), dtype=float)

    # P_i
    paths = nx.shortest_path(G, source=0)
    for i in range(1, n+1):
        for j in range(i, n+1):
            intersect = get_intersecting_path(paths[i], paths[j])
            R[i, j] = sum(r[e] for e in intersect)
            X[i, j] = sum(x[e] for e in intersect)
            R[j, i] = R[i, j]
            X[j, i] = X[i, j]

    R = 2 * R[1:, 1:]
    X = 2 * X[1:, 1:]

    assert is_pos_def(R)
    assert is_pos_def(X)
    return R, X


def calc_voltage_profile(X, R, p, qe, qc, v_sub) -> np.ndarray:
    """Calculates the voltage profile using the simplified linear model.

    Args
    - X: np.array, shape [n, n], positive definite and entry-wise positive
    - R: np.array, shape [n, n], positive definite and entry-wise positive
    - p: np.array, shape [n, T]
    - qe: np.array, shape [n, T]
    - qc: np.array, shape [n, T]
    - v_sub: float, fixed squared voltage magnitude (kV^2) at substation

    Returns
    - v: np.array, shape [n, T]
    """
    return X @ (qc + qe) + R @ p + v_sub

# # To add a new cell, type '# %%'
# # To add a new markdown cell, type '# %% [markdown]'
# # %%
# from __future__ import annotations
# # from IPython import get_ipython

# # %% [markdown]
# # ## Imports and Constants
# # 
# # $$
# # \newcommand{\abs}[1]{\left\lvert#1\right\rvert}  % absolute value
# # \newcommand{\C}{\mathbb{C}}  % complex numbers
# # \newcommand{\ceil}[1]{\left\lceil #1 \right\rceil}  % ceiling
# # \newcommand{\closure}{\operatorname{cl}}  % distance
# # \newcommand{\conv}{\operatorname{conv}}  % convex hull
# # \newcommand{\cov}{\operatorname{Cov}}  % covariance
# # \newcommand{\diag}{\operatorname{diag}}  % diagonal
# # \newcommand{\dist}{\operatorname{dist}}  % distance
# # \newcommand{\dom}{\operatorname{dom}}  % domain
# # \newcommand{\E}{\mathbb{E}}  % expectation
# # \newcommand{\epi}{\operatorname{epi}}  % epigraph
# # \newcommand{\extreme}{\operatorname{extreme}}  % extreme point
# # \newcommand{\floor}[1]{\left\lfloor #1 \right\rfloor}  % floor
# # \newcommand{\inner}[2]{\left\langle#1,\ #2\right\rangle}  % inner product
# # \newcommand{\interior}{\operatorname{int}}  % interior
# # \newcommand{\norm}[1]{\left\lVert#1\right\rVert}  % norm
# # \newcommand{\nullspace}{\operatorname{Null}}  % nullspace
# # \newcommand{\one}{\mathbf{1}}  % ones vector
# # \newcommand{\Proj}{\mathcal{P}}  % projection
# # \newcommand{\range}{\operatorname{Range}}  % range
# # \newcommand{\rank}{\operatorname{rank}}  % rank
# # \newcommand{\set}[1]{\left\{#1\right\}}  % set
# # \newcommand{\spanset}{\operatorname{span}}  % span
# # \newcommand{\Sym}{\mathbb{S}}  % symmetric, real matrices
# # \newcommand{\tr}{\operatorname{tr}}  % trace
# # \newcommand{\var}{\operatorname{Var}}  % variance
# # \newcommand{\zero}{\mathbf{0}}  % ones vector
# # \renewcommand{\N}{\mathbb{N}}  % natural numbers
# # \renewcommand{\R}{\mathbb{R}}  % real numbers
# # \renewcommand{\Z}{\mathbb{Z}}  % integers
# # $$

# # %%
# # get_ipython().run_line_magic('load_ext', 'autoreload')
# # get_ipython().run_line_magic('autoreload', '2')
# # get_ipython().run_line_magic('matplotlib', 'inline')


# # %%
# from collections.abc import Sequence
# import csv
# import time
# from typing import Any, TypeVar

# import cvxpy as cp
# import matplotlib.pyplot as plt
# import networkx as nx
# import numpy as np
# import pandas as pd
# import pandapower as pp
# import pandapower.plotting
# import pandapower.topology
# import scipy.io
# from tqdm.auto import tqdm

# from voltage_control_mpc import (
#     create_R_X,
#     robust_mpc_known_parameters_with_data
# )

# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('svg')

# # hide top and right splines on plots
# plt.rcParams['axes.spines.right'] = False
# plt.rcParams['axes.spines.top'] = False


# # %%
# rng = np.random.default_rng()

# # %% [markdown]
# # ## Setup
# # 
# # Data (TODO: check units)
# # - $r$, $x$: resistance, in ohms
# # - $p$: active power injection, in MW
# # - $q$: reactive power injection, in MVar
# # 
# # The nominal voltage is 12kV.
# # 
# # We know that $P = I V = V^2 / R$. In units, we have
# # $$ W = V^2 / R $$
# # $$ MW = (kV)^2 / R $$

# # %%
# def create_56bus() -> pp.pandapowerNet:
#     """
#     Creates the SCE 56-bus network from the MATPOWER file.
#     Bus 0 is the substation, and the other buses are numbered 1-55.
#     At every bus (except 0), we attach a load and static generator element.

#     Returns: pp.pandapowerNet
#     """
#     net = pp.converter.from_mpc('data/SCE_56bus.mat', casename_mpc_file='case_mpc')

#     # remove loads and generators at all buses except bus 0 (substation),
#     # but keep the network lines
#     buses = list(range(1, 56))
#     pp.drop_elements_at_buses(net, buses=buses, bus_elements=True, branch_elements=False)

#     for i in buses:
#         pp.create_load(net, bus=i, p_mw=0, q_mvar=0)
#         pp.create_sgen(net, bus=i, p_mw=0, q_mvar=0)

#     return net


# # %%
# net = create_56bus()

# fig, ax = plt.subplots(1,1)
# pp.plotting.simple_plot(net, ax=ax)


# # %%
# # read in r and x matrices from data
# n = len(net.bus) - 1  # number of buses, excluding substation
# r = np.ones((n+1, n+1)) * np.inf
# x = np.ones((n+1, n+1)) * np.inf

# r[net.line['from_bus'], net.line['to_bus']] = net.line['r_ohm_per_km']
# r[net.line['to_bus'], net.line['from_bus']] = net.line['r_ohm_per_km']
# x[net.line['from_bus'], net.line['to_bus']] = net.line['x_ohm_per_km']
# x[net.line['to_bus'], net.line['from_bus']] = net.line['x_ohm_per_km']

# G = pp.topology.create_nxgraph(net)
# R, X = create_R_X(r, x, G)

# # %% [markdown]
# # $$w = \underbrace{R (p_t - p_{t-1})}_{w^p} + \underbrace{X (q_t - q_{t-1})}_{w^q}$$

# # %%
# def read_load_data() -> tuple[np.array, np.array]:
#     """Read in load data.

#     Returns
#     - p: np.array, shape [n, T], active load in MW, TODO sign
#     - q: np.array, shape [n, T], reactive load in MVar, TODO sign
#     """
#     mat = scipy.io.loadmat('data/pq_fluc.mat', squeeze_me=True)
#     pq_fluc = mat['pq_fluc']  # shape (55, 2, 14421)
#     p = pq_fluc[:, 0]  # active load, shape (55, 14421)
#     qe = pq_fluc[:, 1]  # reactive load
#     return p, qe

# def smooth(x: np.ndarray, w: int = 5) -> np.ndarray:
#     """Smooths input using moving-average window.

#     Edge values are preserved as-is without smoothing.

#     Args
#     - x: np.array, shape [T] or [n, T]
#     - w: int, moving average window, odd positive integer

#     Returns: np.array, same shape as x, smoothed
#     """
#     assert w % 2 == 1
#     edge = w // 2

#     x_smooth = x.copy()
#     ones = np.ones(w)
#     if len(x.shape) == 1:
#         x_smooth[edge:-edge] = np.convolve(x, ones, 'valid') / w
#     elif len(x.shape) == 2:
#         for i in range(len(x)):
#             x_smooth[i, edge:-edge] = np.convolve(x[i], ones, 'valid') / w
#     else:
#         raise ValueError('smooth() only works on 1D or 2D arrays')
#     return x_smooth

# def calc_max_norm_w(R, X, p, qe):
#     """Calculates ||w||_∞.

#     Args
#     - R: shape [n, n]
#     - X: shape [n, n]
#     - p: shape [n, T], active power load
#     - qe: shape [n, T], exogenous reactive load

#     Returns
#     - norms: dict, keys are ['w', 'wp', 'wq']
#     - max_p_idx: int, bus index with largest ||w_p||
#     - max_q_idx: int, bus index with largest ||w_q||
#     """
#     wp = R @ (p[:, 1:] - p[:, :-1])
#     wq = X @ (qe[:, 1:] - qe[:, :-1])
#     w = wp + wq
#     norms = {
#         'w':  np.linalg.norm( w, ord=np.inf, axis=0),
#         'wp': np.linalg.norm(wp, ord=np.inf, axis=0),
#         'wq': np.linalg.norm(wq, ord=np.inf, axis=0)
#     }
#     max_p_idx = np.argmax(np.max(np.abs(wp), axis=1))
#     max_q_idx = np.argmax(np.max(np.abs(wq), axis=1))
#     return norms, max_p_idx, max_q_idx


# # %%
# p, qe = read_load_data()  # in MW and MVar (TODO: confirm units)

# # before smoothing
# norms, max_p_idx, max_q_idx = calc_max_norm_w(R, X, p, qe)

# fig, axs = plt.subplots(2, 2, figsize=(10, 4), sharex=True, tight_layout=True, gridspec_kw={'height_ratios': [2, 1]})
# for i in range(len(p)):
#     axs[0, 0].plot(p[i])
#     axs[0, 1].plot(qe[i])
# axs[0, 0].set(ylabel='active power (MW)')  # TODO: units
# axs[0, 1].set(ylabel='reactive power (MVar)')  # TODO: units
# axs[1, 0].plot(np.diff(p[max_p_idx]), label=f'bus {max_p_idx}')
# axs[1, 1].plot(np.diff(qe[max_q_idx]), label=f'bus {max_q_idx}')
# axs[1, 0].set(xlabel='time $t$', ylabel='$p_{t+1} - p_t$ (MW)')
# axs[1, 1].set(xlabel='time $t$', ylabel='$q^e_{t+1} - q^e_t$ (MVar)')
# axs[1, 0].legend()
# axs[1, 1].legend()
# fig.suptitle('No smoothing')

# fig, axs = plt.subplots(1, 3, figsize=(10, 2), sharey=True, tight_layout=True)
# axs[0].hist(norms['w'], bins=200)
# axs[0].set(xlabel='$||w||_\infty$', ylabel='count', yscale='log')
# axs[1].hist(norms['wp'], bins=200)
# axs[1].set(xlabel='$||w^p||_\infty$')
# axs[2].hist(norms['wq'], bins=200)
# axs[2].set(xlabel='$||w^q||_\infty$')
# fig.suptitle('No smoothing')
# plt.show()

# # after smoothing p's
# for w in range(3, 51, 2):
#     norms, max_p_idx, max_q_idx = calc_max_norm_w(R, X, smooth(p, w=w), qe)
#     max_norm = norms['w'].max()
#     if max_norm <= 0.7:
#         print(f'w = {w}, max ||w|| = {max_norm}')
#         break
# p = smooth(p, w=w)

# fig, axs = plt.subplots(2, 1, figsize=(5, 4), sharex=True, tight_layout=True, gridspec_kw={'height_ratios': [2, 1]})
# for i in range(len(p)):
#     axs[0].plot(p[i])
# axs[0].set(xlabel='time $t$', ylabel='active power (MW)')  # TODO: units
# axs[1].plot(np.diff(p[max_p_idx]), label=f'bus {max_p_idx}')
# axs[1].set(xlabel='time $t$', ylabel='$p_{t+1} - p_t$')
# axs[1].legend()
# fig.suptitle(f'After smoothing ($w={w}$)')

# fig, axs = plt.subplots(1, 3, figsize=(10, 2), sharey=True, tight_layout=True)
# axs[0].hist(norms['w'], bins=200)
# axs[0].set(xlabel='$||w||_\infty$', ylabel='count', yscale='log')
# axs[1].hist(norms['wp'], bins=200)
# axs[1].set(xlabel='$||w^p||_\infty$')
# axs[2].hist(norms['wq'], bins=200)
# axs[2].set(xlabel='$||w^q||_\infty$')
# fig.suptitle(f'After smoothing ($w={w}$)')
# plt.show()

# # %% [markdown]
# # ## Robust MPC for Voltage Control with Known Line Parameters
# # 
# # If the line parameters $X$ are known, then we can directly apply robust MPC, where the robustness is with respect to the noise in the system dynamics. To guarantee convergence by time $T_f$, we adopt a shrinking horizon once the current time $t$ exceeds $T_f - T$, where $T$ is our MPC optimization horizon.
# # 
# # $$
# # \begin{aligned}
# # \min_{u_i \in \R^n,\ \forall i \in \{0, \dotsc, T-1\}} \quad
# #     & \sum_{t=0}^{T-1} u_t^\top P u_t \\
# # \text{s.t.} \quad
# #     & \underline{u} \leq u_t \leq \overline{u}, &&\forall t = 0, \dotsc, T-1 \\
# #     & \underline{q} \leq \sum_{t=0}^i u_t \leq \overline{q}, &&\forall i = 0, \dotsc, T-1 \\
# #     & v_0 + X \sum_{t=0}^{T-1} u_t + \sum_{t=0}^{T-1} w_t \in [\underline{v}, \overline{v}] \\
# #     & \forall w_t: \norm{w_t}_\infty \leq \eta
# # \end{aligned}
# # $$
# # %% [markdown]
# # ### Assuming linear system dynamics
# # 
# # In this section, we assume the Simplified DistFlow linear model.

# # %%
# v_0 = rng.uniform((0.97 * 12) ** 2, (0.98 * 12)**2, size=[n])  # initial voltage
# print(v_0)


# # %%
# eta = 1  # bound on inf-norm of w_t
# T = 8
# T_f = 80


# # %%
# robust_mpc_known_parameters_with_data(
#     v_0=v_0, R=R, X=X, p=p, qe=qe,
#     eta=eta, T=T, T_f=T_f
# )


# # %%
# for T in range(1, 20):
#     try:
#         v_history, u_history, cost = robust_mpc_known_parameters(v_0=v_0, X=X, eta=eta, T=T, T_f=T_f)
#         break
#     except RuntimeError as e:
#         print(e)
#         pass


