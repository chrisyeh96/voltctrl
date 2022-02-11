from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar
import warnings

import networkx as nx
import numpy as np
import pandapower as pp
import pandapower.topology
import scipy.io

warnings.filterwarnings("ignore", category=FutureWarning)

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


def create_RX_from_net(net: pp.pandapowerNet, noise: float = 0, perm: bool = False,
                       seed: int | None = 123, check_pd: bool = True
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Creates R,X matrices from a pandapowerNet.

    Args
    - net: pandapowerNet with (n+1) buses including substation
    - noise: float, optional uniform noise to add to impedances, values in [0,1]
    - seed: int, for generating the uniform noise

    Returns: tuple (X, R)
    - X: np.array, shape [n, n], positive definite and entry-wise positive
    - R: np.array, shape [n, n], positive definite and entry-wise positive
    """
    assert 0 <= noise <= 1, 'noise must be a float in [0,1]'

    # read in r and x matrices from data
    n = len(net.bus) - 1  # number of buses, excluding substation
    r = np.ones((n+1, n+1)) * np.inf
    x = np.ones((n+1, n+1)) * np.inf

    r_ohm_per_km = net.line['r_ohm_per_km'].values
    x_ohm_per_km = net.line['x_ohm_per_km'].values
    from_bus = net.line['from_bus']
    to_bus = net.line['to_bus']

    rng = np.random.default_rng(seed)

    if noise > 0:
        # Do NOT update r/x_ohm_per_km in-place. We do not want to change
        # the underlying net object.
        noise_limit = r_ohm_per_km * noise
        r_ohm_per_km = r_ohm_per_km + rng.uniform(-noise_limit, noise_limit)

        noise_limit = x_ohm_per_km * noise
        x_ohm_per_km = x_ohm_per_km + rng.uniform(-noise_limit, noise_limit)

    if perm:  # permute the line numbers
        order = np.concatenate([[0], rng.permutation(np.arange(1, n+1))])
        net.line['from_bus'] = net.line['from_bus'].map(order.__getitem__)
        net.line['to_bus'] = net.line['to_bus'].map(order.__getitem__)
        # r_ohm_per_km = rng.permutation(r_ohm_per_km)
        # x_ohm_per_km = rng.permutation(x_ohm_per_km)

    r[net.line['from_bus'], net.line['to_bus']] = r_ohm_per_km
    r[net.line['to_bus'], net.line['from_bus']] = r_ohm_per_km
    x[net.line['from_bus'], net.line['to_bus']] = x_ohm_per_km
    x[net.line['to_bus'], net.line['from_bus']] = x_ohm_per_km

    G = pp.topology.create_nxgraph(net)
    R, X = create_RX_from_rx(r, x, G, check_pd)
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


def make_pd_and_pos(A: np.ndarray) -> None:
    """
    Tries to make matrix `A` PSD and entrywise positive.
    Updates A in-place.
    Guarantees output to be PSD. Does NOT guarantee entrywise positive.
    """
    A[:] = (A + A.T) / 2  # make symmetric
    np.maximum(0, A, out=A)  # make positive, in-place
    w, V = np.linalg.eigh(A)
    if w[0] < 0:
        w[w < 0] = 1e-7
        A[:] = (V * w) @ V.T


def create_RX_from_rx(r: np.ndarray, x: np.ndarray, G: nx.Graph,
                      check_pd: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Creates R,X matrices from line impedance matrices r and x.

    Args
    - r: np.array, shape [n+1, n+1], symmetric and entry-wise positive
    - x: np.array, shape [n+1, n+1], symmetric and entry-wise positive
    - G: nx.Graph, undirected graph

    Returns: tuple (X, R)
    - X: np.array, shape [n, n], positive definite and entry-wise positive
    - R: np.array, shape [n, n], positive definite and entry-wise positive
    """
    n = r.shape[0] - 1

    R = np.zeros((n+1, n+1), dtype=float)
    X = np.zeros((n+1, n+1), dtype=float)

    # P_i
    paths = nx.shortest_path(G, source=0)  # node i => path from node 0 to i
    for i in range(1, n+1):
        for j in range(i, n+1):
            intersect = get_intersecting_path(paths[i], paths[j])
            R[i, j] = sum(r[e] for e in intersect)
            X[i, j] = sum(x[e] for e in intersect)
            R[j, i] = R[i, j]
            X[j, i] = X[i, j]

    R = 2 * R[1:, 1:]
    X = 2 * X[1:, 1:]

    assert np.all(R != np.inf)
    assert np.all(X != np.inf)

    if check_pd:
        assert is_pos_def(R)
        assert is_pos_def(X)
    return R, X


def calc_voltage_profile(X: np.ndarray, R: np.ndarray, p: np.ndarray,
                         qe: np.ndarray, qc: np.ndarray, v_sub: float
                         ) -> np.ndarray:
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


def read_load_data() -> tuple[np.ndarray, np.ndarray]:
    """Read in load data.

    Returns
    - p: np.array, shape [T, n], active load in MW, TODO sign
    - q: np.array, shape [T, n], reactive load in MVar, TODO sign
    """
    mat = scipy.io.loadmat('data/pq_fluc.mat', squeeze_me=True)
    pq_fluc = mat['pq_fluc']  # shape (55, 2, 14421)
    p = pq_fluc[:, 0]  # active load, shape (55, 14421)
    qe = pq_fluc[:, 1]  # reactive load
    return p.T, qe.T


def smooth(x: np.ndarray, w: int = 5) -> np.ndarray:
    """Smooths input using moving-average window.

    Edge values are preserved as-is without smoothing.

    Args
    - x: np.array, shape [T] or [n, T]
    - w: int, moving average window, odd positive integer

    Returns: np.array, same shape as x, smoothed
    """
    assert w % 2 == 1
    edge = w // 2

    x_smooth = x.copy()
    ones = np.ones(w)
    if len(x.shape) == 1:
        x_smooth[edge:-edge] = np.convolve(x, ones, 'valid') / w
    elif len(x.shape) == 2:
        for i in range(len(x)):
            x_smooth[i, edge:-edge] = np.convolve(x[i], ones, 'valid') / w
    else:
        raise ValueError('smooth() only works on 1D or 2D arrays')
    return x_smooth


def calc_max_norm_w(R: np.ndarray, X: np.ndarray, p: np.ndarray, qe: np.ndarray
                   ) -> dict[str, np.ndarray]:
    """Calculates ||w||_∞.

    Args
    - R: shape [n, n]
    - X: shape [n, n]
    - p: shape [n, T], active power load
    - qe: shape [n, T], exogenous reactive load

    Returns: norms, dict maps keys ['w', 'wp', 'wq'] to np.ndarray of shape [T]
    """
    wp = R @ (p[:, 1:] - p[:, :-1])
    wq = X @ (qe[:, 1:] - qe[:, :-1])
    w = wp + wq
    norms = {
        'w':  np.linalg.norm( w, ord=np.inf, axis=0),
        'wp': np.linalg.norm(wp, ord=np.inf, axis=0),
        'wq': np.linalg.norm(wq, ord=np.inf, axis=0)
    }
    # - max_p_idx: int, bus index with largest ||w_p||
    # - max_q_idx: int, bus index with largest ||w_q||
    # max_p_idx = np.argmax(np.max(np.abs(wp), axis=1))
    # max_q_idx = np.argmax(np.max(np.abs(wq), axis=1))
    return norms
