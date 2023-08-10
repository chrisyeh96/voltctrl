from __future__ import annotations

from collections.abc import MutableMapping, MutableSet, Sequence
import copy
from typing import TypeVar
import warnings

import cvxpy as cp
import networkx as nx
import numpy as np
import pandapower as pp
import pandapower.topology
import scipy.io

warnings.filterwarnings('ignore', category=FutureWarning)

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
    pp.drop_elements_at_buses(
        net, buses=buses, bus_elements=True, branch_elements=False)

    for i in buses:
        pp.create_load(net, bus=i, p_mw=0, q_mvar=0)
        pp.create_sgen(net, bus=i, p_mw=0, q_mvar=0)

    return net


def create_RX_from_net(net: pp.pandapowerNet, noise: float = 0,
                       modify: str | None = None, seed: int | None = 123,
                       check_pd: bool = True
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Creates R,X matrices from a pandapowerNet.

    Args
    - net: pandapowerNet with (n+1) buses including substation
    - noise: float, optional add uniform noise to impedances, values in [0,1]
    - modify: str, how to modify network, one of [None, 'perm', 'linear', 'rand']
    - seed: int, for generating the uniform noise
        seed must be provided if (noise > 0) or (modify is not None)
    - check_pd: bool, whether to assert that returned R,X are PD

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

    if seed is not None:
        rng = np.random.default_rng(seed)

    if noise > 0:
        # Do NOT update r/x_ohm_per_km in-place. We do not want to change
        # the underlying net object.
        noise_limit = r_ohm_per_km * noise
        r_ohm_per_km = r_ohm_per_km + rng.uniform(-noise_limit, noise_limit)

        noise_limit = x_ohm_per_km * noise
        x_ohm_per_km = x_ohm_per_km + rng.uniform(-noise_limit, noise_limit)

    if modify in ('perm', None):  # permute the line numbers
        net = copy.deepcopy(net)  # don't modify original net
        if modify == 'perm':
            order = np.zeros(n+1, dtype=int)
            order[1:] = rng.permutation(np.arange(1, n+1))
            net.line['from_bus'] = net.line['from_bus'].map(order.__getitem__)
            net.line['to_bus'] = net.line['to_bus'].map(order.__getitem__)

        r[net.line['from_bus'], net.line['to_bus']] = r_ohm_per_km
        r[net.line['to_bus'], net.line['from_bus']] = r_ohm_per_km
        x[net.line['from_bus'], net.line['to_bus']] = x_ohm_per_km
        x[net.line['to_bus'], net.line['from_bus']] = x_ohm_per_km
        G = pp.topology.create_nxgraph(net)

    elif modify in ('linear', 'rand'):
        if modify == 'linear':  # random undirected linear tree
            # substation (node 0) is not necessarily at one end of the path,
            # could be in the middle
            path = rng.permutation(n+1)
            G = nx.path_graph(path)
        else:
            G = nx.random_tree(n+1)  # uniformly random undirected tree

        r_sample = rng.choice(r_ohm_per_km, size=len(G.edges), replace=True)
        x_sample = rng.choice(x_ohm_per_km, size=len(G.edges), replace=True)
        for i, (e0, e1) in enumerate(G.edges):
            r[e0, e1] = r_sample[i]
            x[e0, e1] = x_sample[i]
            r[e1, e0] = r[e0, e1]
            x[e1, e0] = x[e0, e1]
    else:
        raise ValueError(f'Unexpected value for `modify`: {modify}')

    R, X = create_RX_from_rx(r, x, G, check_pd)
    return R, X


def get_intersecting_path(path1: Sequence[T], path2: Sequence[T]
                          ) -> list[tuple[T, T]]:
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
    if not np.array_equal(A, A.T):
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
    - G: nx.Graph, undirected graph, nodes are numbered {0, ..., n}
    - check_pd: bool, whether to assert that returned R,X are PD

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
    """Read in power injection data.

    Returns
    - p: np.array, shape [T, n], net active power injection in MW
    - q: np.array, shape [T, n], exogenous reactive power injection in MVar
    """
    mat = scipy.io.loadmat('data/pq_fluc.mat', squeeze_me=True)
    pq_fluc = mat['pq_fluc']  # shape (55, 2, 14421)
    p = pq_fluc[:, 0]  # net active power injection, shape (55, 14421)
    qe = pq_fluc[:, 1]  # exogenous reactive power injection
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
    """Calculates ‖w‖_∞.

    Args
    - R: np.array, shape [n, n]
    - X: np.array, shape [n, n]
    - p: np.array, shape [n, T], active power injection
    - qe: np.array, shape [n, T], exogenous reactive power injection

    Returns: norms, dict maps keys ['w', 'wp', 'wq'] to np.ndarray of shape [T]
    """
    wp = R @ (p[:, 1:] - p[:, :-1])
    wq = X @ (qe[:, 1:] - qe[:, :-1])
    w = wp + wq
    norms = {
        'w':  np.linalg.norm( w, ord=np.inf, axis=0),  # noqa
        'wp': np.linalg.norm(wp, ord=np.inf, axis=0),
        'wq': np.linalg.norm(wq, ord=np.inf, axis=0)
    }
    # - max_p_idx: int, bus index with largest ‖w_p‖
    # - max_q_idx: int, bus index with largest ‖w_q‖
    # max_p_idx = np.argmax(np.max(np.abs(wp), axis=1))
    # max_q_idx = np.argmax(np.max(np.abs(wq), axis=1))
    return norms


def np_triangle_norm(x: np.ndarray) -> float:
    """Computes ‖X‖_△"""
    return float(np.linalg.norm(np.triu(x), ord='fro'))


def known_topology_constraints(
        X: cp.Variable,
        net: pp.pandapowerNet,
        known_line_params: int,
        known_bus_topo: int
        ) -> list[cp.Constraint]:
    """Specifies constraints on X matrix if we know the network topology
    among all buses in {1, ..., known_bus_topo}.

    Args
    - X: shape [n, n], optimization variable
    - net: pandapowerNet representing a tree-structured distribution grid
        with (n+1) buses numbered [0 (substation), ..., n] such that if
        bus i is a parent of bus j, then i < j
    - known_bus_topo: int in [0, n], n = # of buses (excluding substation),
        when topology is known for buses/lines in {1, ..., known_bus_topo}
    - known_line_params: int in [0, known_bus_topo], when line parameters
        (little x_{ij}) are known ∀ i,j in {1, ..., known_line_params}

    Returns: list of cp.Constraints
    """
    assert known_bus_topo >= 0
    if known_bus_topo == 0:
        return []

    G = pp.topology.create_nxgraph(net)  # buses numbered 0, ..., n+1
    mapping = {i: i-1 for i in range(len(G))}
    nx.relabel_nodes(G, mapping=mapping, copy=False)  # buses numbered -1, 0, ..., n
    DG = nx.bfs_tree(G, source=-1)  # tree, edges parent -> child

    for n1, n2 in DG.edges:
        assert n1 < n2

    constraints = []
    for i in range(known_bus_topo):
        for j in range(i, known_bus_topo):
            # if the line params are known, then those constraints will
            # supersede the topology constraints
            if i < known_line_params and j < known_line_params:
                continue

            # buses are numbered such that parent(i) < i, so we know the
            # topology relationship between parent and bus i
            # - but bus 0's parent is the substation (-1), and the constraint
            #   X[0, 0] >= 0 is already part of the consistent set definition
            elif i == j:
                if i == 0:
                    continue
                else:
                    parent = next(DG.predecessors(i))
                    constr = (X[i, i] >= X[parent, parent])

            # j > i
            else:
                lca = nx.lowest_common_ancestor(DG, i, j)
                assert lca is not None
                if lca == -1:  # lca is substation
                    constr = (X[i, j] == 0)
                else:
                    constr = (X[i, j] == X[lca, lca])
            constraints.append(constr)
    return constraints


def X_to_ancestors(X: np.ndarray) -> tuple[dict[int, set[int]], np.ndarray]:
    """Constructs ancestor map from X matrix.

    Args
    - X: np.ndarray, shape [n, n], satisfies constraints:
        - PSD, elementwise >= 0, diagonal entries are largest in each row/col

    Returns: set, maps each node to a set of ancestors
    """
    n = X.shape[0]

    # get a list of bins, centered at values along diag(X)
    centers = np.concatenate([[0], np.sort(np.diag(X))])
    bins = [0] + list((centers[:-1] + centers[1:]) / 2)

    # bin the values of X, then replace with bin center
    inds = np.digitize(X, bins) - 1  # digitize returns 1-indexed bin indices
    X = centers[inds.flatten()].reshape(X.shape)

    # create a mapping from nodes => set of ancestors
    # - every node has the substation (node: -1) as an ancestor
    ancestors = {i: {-1} for i in range(n)}

    for i in range(n):
        for j in range(i+1, n):
            if X[i, j] == 0:
                # only common ancestor is the substation
                pass

            elif X[i, j] == X[i, i]:
                # j is a descendant of i
                ancestors[j].add(i)

                # ensure that descendants d of j have X_{id} == X_{ii}
                for d in range(n):
                    if X[j, d] == X[j, j]:
                        X[i, d] = X[i, i]
                        X[d, i] = X[i, i]

            elif X[i, j] == X[j, j]:
                # i is a descendant of j
                ancestors[i].add(j)

                # ensure that descendants d of i have X_{jd} == X_{jj}
                for d in range(n):
                    if X[i, d] == X[i, i]:
                        X[j, d] = X[j, j]
                        X[d, j] = X[j, j]

            else:
                # i,j share a common ancestor k (other than the substation)

                k = int(np.argmin(np.abs(X[i, j] - np.diag(X))))
                assert X[i, j] == X[k, k]
                ancestors[i].add(k)
                ancestors[j].add(k)

                X[j, k] = X[k, j] = X[k, k]

                # ensure that descendants di of i and dj of j have X[di, dj] = shared
                # for di in range(n):
                #     if X[i, di] == X[i, i]:
                #         for dj in range(n):
                #             if X[j, dj] == X[j, j]:
                #                 X[di, dj] = shared
                #                 X[dj, di] = shared

    return ancestors, X


def check_ancestors_completeness(ancestors):
    # complete = {}
    # for n in ancestors:
    #     done = set()
    #     queue = ancestors[n]
    #     while len(queue) > 0:
    #         a = queue.pop()
    #         done.add(a)
    #         if a in complete:
    #             done |= complete[a]
    #         elif a != -1:
    #             queue |= (ancestors[a] - done)
    #     complete[n] = done

    for n in ancestors:
        for a in ancestors[n]:
            if a != -1:
                assert ancestors[a] <= ancestors[n]


def build_tree_from_ancestors(ancestors: MutableMapping[int, MutableSet[int]]
                              ) -> nx.Graph:
    """Builds tree from ancestors mapping.

    Args
    - ancestors: set, maps each node to a set of ancestors

    Notes:
    - Assumes that ancestors mapping forms a DAG instead of a tree. To create
        a tree from a DAG, each child node randomly picks one of its parents
        to be its single parent.
    - See https://cs.stackexchange.com/q/23408

    Returns: nx.Graph, tree structure
    """
    ancestors = copy.deepcopy(ancestors)
    G = nx.Graph()
    while len(ancestors) > 0:
        # find all nodes with only 1 ancestor
        children = set()
        remaining = set(ancestors.keys())
        for n in ancestors:
            if len(ancestors[n]) == 1 or ancestors[n].isdisjoint(remaining):
                parent = next(iter(ancestors[n]))  # choose a parent at random
                children.add(n)
                G.add_edge(parent, n)

        if len(children) == 0:
            raise ValueError('Invalid ancestors map')

        for n in ancestors:
            for c in children:
                if c in ancestors[n]:
                    ancestors[n] -= ancestors[c]
        for c in children:
            del ancestors[c]
    return G
