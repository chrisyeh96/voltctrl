from __future__ import annotations

from collections.abc import Sequence
import pickle
import datetime as dt
import os
from typing import Any

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from cbc.base import (
    CBCBase, CBCConst, CBCConstWithNoise, project_into_X_set)
from network_utils import (
    create_56bus, create_RX_from_net, meta_gen_X_set, read_load_data)
from robust_voltage_control_nonlinear import (
    VoltPlot, robust_voltage_control)
from utils import wrap_write_newlines

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# hide top and right splines on plots
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False


def run(ε: float, q_max: float, cbc_alg: str, eta: float,
        norm_bound: float, norm_bound_init: float | None = None,
        noise: float = 0, modify: str | None = None, δ: float = 0.,
        obs_nodes: Sequence[int] | None = None,
        ctrl_nodes: Sequence[int] | None = None,
        known_bus_topo: int = 0, known_line_params: int = 0,
        nsamples: int = 100, seed: int = 123,
        is_interactive: bool = False, savedir: str = '',
        pbar: tqdm | None = None,
        tag: str = '') -> str:
    """
    Args
    - ε: float, robustness
    - q_max: float, maximum reactive power injection
    - cbc_alg: str, one of ['const', 'proj', 'steiner']
    - eta: float, maximum ‖w‖∞
    - norm_bound: float, size of uncertainty set
    - norm_bound_init: float or None, norm of uncertainty set from which
        X_init is sampled
    - noise: float, network impedances modified by fraction Uniform(±noise)
    - modify: str, how to modify network, one of [None, 'perm', 'linear', 'rand']
    - δ: float, weight of noise term in CBC norm when learning eta
    - obs_nodes: list of int, nodes that we can observe voltages for,
        set to None if we observe all voltages
    - ctrl_nodes: list of int, nodes that we can control voltages for,
        set to None if we control all voltages
    - known_bus_topo: int in [0, n], n = # of buses (excluding substation),
        when topology is known for buses/lines in {0, ..., known_bus_topo-1}
    - known_line_params: int in [0, known_bus_topo], when line parameters
        (little x_{ij}) are known ∀ i,j in {0, ..., known_line_params-1}
    - nsamples: int, # of samples to use for computing consistent set,
        only used when cbc_alg is 'proj' or 'steiner'
    - seed: int, random seed
    - is_interactive: bool, whether to output to screen, or log to disk
    - savedir: str, path to folder for saving outputs ('' for current dir)
    - pbar: tqdm instance
    - tag: str, arbitrary tag to add to filename ('' for no tag)

    Returns: str, filename (without extension)
    """
    assert δ >= 0, 'δ must be >= 0'

    if savedir != '':
        os.makedirs(savedir, exist_ok=True)
    tz = dt.timezone(dt.timedelta(hours=-8))  # PST
    start_time = dt.datetime.now(tz)

    config: dict[str, Any] = dict(
        cbc_alg=cbc_alg, q_max=q_max, ε=ε, eta=eta, δ=δ,
        obs_nodes=obs_nodes, ctrl_nodes=ctrl_nodes, seed=seed,
        known_bus_topo=known_bus_topo, known_line_params=known_line_params)
    filename = os.path.join(savedir, f'CBC{cbc_alg}')

    if δ > 0:
        filename += f'_δ{δ}_η{eta}'

    # read in data
    if noise > 0 or modify is not None:
        config.update(norm_bound=norm_bound, norm_bound_init=norm_bound_init)
        if noise > 0:
            config.update(noise=noise)
            filename += f'_noise{noise}'
        if modify is not None:
            config.update(modify=modify)
            filename += f'_{modify}'
        if norm_bound_init is not None:
            filename += f'_norminit{norm_bound_init}'
        filename += f'_norm{norm_bound}_seed{seed}'

    net = create_56bus()
    R, X = create_RX_from_net(net, noise=0)  # true R and X
    p, qe = read_load_data()  # in MW and MVar
    T, n = p.shape

    # ==== FIXED PARAMETERS ====
    v_min, v_max = (11.4**2, 12.6**2)  # +/-5%, units kV^2
    v_nom = 12**2  # nominal squared voltage magnitude, units kV^2
    v_sub = v_nom  # fixed squared voltage magnitude at substation, units kV^2

    Pv = 0.1
    Pu = 10

    # weights on slack variables: alpha for CBC, β for robust oracle
    alpha = 1000  # only used when not learning eta, set to 0 to turn off slack variable
    β = 100

    config.update(v_min=v_min, v_max=v_max, v_nom=v_nom, Pv=Pv, Pu=Pu, β=β)
    # ==== end of FIXED PARAMETERS ====

    filename += tag
    filename += start_time.strftime('_%Y%m%d_%H%M%S')
    if is_interactive:
        log = tqdm
    else:
        log = wrap_write_newlines(open(f'{filename}.log', 'w'))
        print(f'filename: {filename}')
    log.write(f'filename: {filename}')

    # ==== NONLINEAR MODIFICATIONS ====
    # Load nonlinear vpar (i.e., nonlinear power flow voltage without control)
    vpars = np.load('data/nonlinear_voltage_baseline.npy')[:, 1:]  # shape [T, n]
    vpars = (vpars * 12.)**2
    vpar_min = np.min(vpars, axis=0) - 0.5
    vpar_max = np.max(vpars, axis=0) + 0.5
    Vpar = (vpar_min, vpar_max)
    # ==== NONLINEAR MODIFICATIONS ====

    start = 0  # starting time step

    # randomly initialize a network matrix
    _, X_init = create_RX_from_net(net, noise=noise, modify=modify,
                                   check_pd=True, seed=seed)
    save_dict = dict(X_init=X_init)
    if norm_bound_init is not None:
        assert norm_bound_init < norm_bound
        var_X = cp.Variable(X.shape, PSD=True)
        init_X_set = meta_gen_X_set(
            norm_bound=norm_bound_init, X_true=X, net=net,
            known_bus_topo=known_bus_topo, known_line_params=known_line_params
        )(var_X)
        project_into_X_set(X_init=X_init, var_X=var_X, log=log,
                           X_set=init_X_set, X_true=X)
        X_init = var_X.value

    gen_X_set = meta_gen_X_set(
        norm_bound=norm_bound, X_true=X, net=net,
        known_bus_topo=known_bus_topo, known_line_params=known_line_params)

    sel: CBCBase
    require_X_psd = True
    if cbc_alg == 'const':
        if δ == 0:
            sel = CBCConst(
                n=n, T=T, X_init=X_init, v=vpars[start],
                gen_X_set=gen_X_set, X_true=X, obs_nodes=obs_nodes, log=log)
        else:
            sel = CBCConstWithNoise(
                n=n, T=T, X_init=X_init, v=vpars[start],
                gen_X_set=gen_X_set, X_true=X, obs_nodes=obs_nodes, log=log)
    elif cbc_alg == 'lsq':
        assert δ == 0
        from cbc.lsq import CBCLsq
        sel = CBCLsq(
            n=n, T=T, X_init=X_init, v=vpars[start],
            gen_X_set=gen_X_set, X_true=X, obs_nodes=obs_nodes, log=log)
        require_X_psd = False
    elif cbc_alg == 'proj':
        from cbc.projection import CBCProjection, CBCProjectionWithNoise
        config.update(alpha=alpha, nsamples=nsamples)
        if δ == 0:
            sel = CBCProjection(
                n=n, T=T-start, X_init=X_init, v=vpars[start],
                gen_X_set=gen_X_set, eta=eta, nsamples=nsamples, alpha=alpha,
                Vpar=Vpar, X_true=X, obs_nodes=obs_nodes, log=log, seed=seed)
        else:
            sel = CBCProjectionWithNoise(
                n=n, T=T-start, X_init=X_init, eta_init=0, v=vpars[start],
                gen_X_set=gen_X_set, eta=eta, nsamples=nsamples, δ=δ,
                Vpar=Vpar, X_true=X, obs_nodes=obs_nodes, log=log, seed=seed)
        # save_dict.update(w_inds=sel.w_inds, vpar_inds=sel.vpar_inds)
    elif cbc_alg == 'steiner':
        assert δ == 0
        from cbc.steiner import CBCSteiner
        dim = n * (n+1) // 2
        config.update(nsamples=nsamples, nsamples_steiner=dim)
        sel = CBCSteiner(
            eta=eta, n=n, T=T-start, nsamples=nsamples, nsamples_steiner=dim,
            v=vpars[start], gen_X_set=gen_X_set, Vpar=Vpar,
            X_init=X_init, X_true=X, obs_nodes=obs_nodes, log=log, seed=seed)
    else:
        raise ValueError('unknown cbc_alg')

    volt_plot = VoltPlot(
        v_lims=(np.sqrt(v_min), np.sqrt(v_max)),
        q_lims=(-q_max, q_max))

    vs, qcs, dists, params, check_prediction = robust_voltage_control(
        p=p[start:T], qe=qe[start:T],
        v_lims=(v_min, v_max), q_lims=(-q_max, q_max), v_nom=v_nom, net=net,
        X=X, R=R, require_X_psd=require_X_psd, Pv=Pv * np.eye(n), Pu=Pu * np.eye(n),
        eta=eta, ε=ε, v_sub=v_sub, β=β, sel=sel, δ=δ,
        ctrl_nodes=ctrl_nodes, pbar=pbar, log=log,
        volt_plot=volt_plot if is_interactive else None)

    elapsed = (dt.datetime.now(tz) - start_time).total_seconds()

    # save data
    with open(f'{filename}.pkl', 'wb') as f:
        pickle.dump(file=f, obj=dict(
            vs=vs, qcs=qcs, dists=dists, params=params, config=config, pred_error=check_prediction,
            elapsed=elapsed, **save_dict))

    # plot and save figure
    volt_plot.update(qcs=qcs,
                     vs=np.sqrt(vs),
                     vpars=np.sqrt(vpars),
                     dists=(dists['t'], dists['X_true']))
    volt_plot.fig.savefig(f'{filename}.svg', pad_inches=0, bbox_inches='tight')
    volt_plot.fig.savefig(f'{filename}.pdf', pad_inches=0, bbox_inches='tight')

    # plot and save voltage prediction error for proposed algorithm and fixed linearization
    # fig, ax = plt.subplots()
    # for name in check_prediction:
    #     ax.plot(np.arange(len(check_prediction[name])), np.cumsum(np.array(check_prediction[name])), label=name)
    # ax.legend()
    # fig.savefig(f'{filename}_prediction.png')
    # fig.show()

    if not is_interactive:
        log.close()
    return filename


if __name__ == '__main__':
    savedir = 'out/nonlinear'
    # all_nodes = np.arange(55)
    # exclude = np.array([8, 18, 21, 30, 39, 45, 54]) - 1
    # obs_nodes = np.setdiff1d(all_nodes, exclude).tolist()
    obs_nodes = None
    for seed in [8, 9, 10, 11]:
        run(
            ε=0.1,
            q_max=0.24,
            cbc_alg='proj',
            eta=11.5,  # 8.65,
            norm_bound=1.0,
            norm_bound_init=None,
            noise=1.0,
            modify='perm',
            δ=20,
            obs_nodes=obs_nodes,
            ctrl_nodes=obs_nodes,
            known_line_params=14,
            known_bus_topo=14,
            seed=seed,
            pbar=tqdm(),
            is_interactive=False,
            savedir=savedir,
            tag='_knownlines14')  # choose from ['', '_partialobs', '_partialctrl', '_knowntopoX', '_knownlinesX']

    # fixed X*, known eta
    # run(
    #     ε=0.1,
    #     q_max=0.24,
    #     cbc_alg='const',
    #     eta=8.65,
    #     norm_bound=0.,
    #     norm_bound_init=None,
    #     noise=0,
    #     modify=None,
    #     δ=0,
    #     obs_nodes=None,
    #     ctrl_nodes=None,
    #     known_line_params=0,
    #     known_bus_topo=0,
    #     seed=None,
    #     pbar=tqdm(),
    #     is_interactive=False,
    #     savedir=savedir,
    #     tag='')

    # fixed X*, unknown eta
    # run(
    #     ε=0.1,
    #     q_max=0.24,
    #     cbc_alg='const',
    #     eta=10,
    #     norm_bound=0.,
    #     norm_bound_init=None,
    #     noise=0,
    #     modify=None,
    #     δ=20,
    #     obs_nodes=None,
    #     ctrl_nodes=None,
    #     known_line_params=0,
    #     known_bus_topo=0,
    #     seed=None,
    #     pbar=tqdm(),
    #     is_interactive=False,
    #     savedir=savedir,
    #     tag='')
