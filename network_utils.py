from __future__ import annotations

from collections.abc import Sequence
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
