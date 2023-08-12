"""
This script creates the following file: data/nonlinear_voltage_baseline.py.

Before running this script, make sure that orig_data.zip has been unzipped
into a folder called orig_data/ located in the root of this repo.
"""
from __future__ import annotations

import numpy as np
import pandapower as pp
import scipy.io as spio
from tqdm.auto import tqdm

from network_utils import create_56bus


n = 55
T = 14421


def read_load_data() -> tuple[np.ndarray, np.ndarray]:
    """Reads in load data.

    Returns:
    - load_p: shape [55, 14421], active load in MW
    - load_q: shape [55, 14421], reactive load in MVar
    """
    # each row in is a node (1 - 55)
    # 6 columns: ['name', 'connectionkW', 'kW', 'pf', 'kVar', 'nameopal']
    scale = 1.1
    load = spio.loadmat('orig_data/loadavail20150908.mat', squeeze_me=True)
    load_p = np.stack(load['Load']['kW']) / 1000  # to MW
    load_p *= scale
    load_q = np.stack(load['Load']['kVar']) / 1000  # to MVar
    load_q *= scale
    return load_p, load_q


def read_solar_data() -> np.ndarray:
    """Reads in solar generation data.

    Returns:
    - gen_p: shape [55, 14421], active power generation in MW
    """
    # see Load_PV_systems_3phase_delta.m
    # - simulate up to 18 nodes with PV
    capacities = np.array([
        9.97, 11.36, 13.53, 6.349206814, 106.142148, 154, 600, 293.54, 66.045,
        121.588489, 12.94935415, 19.35015173, 100, 31.17327501, 13.06234596,
        7.659505852, 100, 700])  # in kW
    capacities /= 1000  # to MW

    # for whatever reason, Guanan scales the capacities by a factor of 7
    # - see line 39 in dynamic_simu_setting_revision_2nd.m
    capacities *= 7

    # see Generate_PV_power.m
    solar_orig = spio.loadmat('orig_data/pvavail20150908_2.mat', squeeze_me=True)
    pv_profile = solar_orig['PVavail'][0]['PVp_6s'] / solar_orig['PVavail'][0]['PVacrate']
    pv = pv_profile * capacities.reshape(-1, 1)  # shape [18, 14421]

    # nodes with PV
    # - Guanan sets substation = bus 1, then other nodes are 1,...,56
    # - I use substation = bus -1, then other nodes are 0,...,54
    pv_bus = np.array([9,12,14,16,19,10,11,13,15,7,2,4,20,23,25,26,32,8]) - 2
    gen_p = np.zeros((n, T))
    gen_p[pv_bus, :] = pv
    return gen_p


def main():
    net = create_56bus()
    load_p, load_q = read_load_data()
    gen_p = read_solar_data()

    v = np.zeros((T, n+1))
    zero_action = np.zeros(n)

    for t in tqdm(range(T)):
        net.load['p_mw'] = load_p[:,t]
        net.load['q_mvar'] = load_q[:,t]
        net.sgen['p_mw'] = gen_p[:,t]
        net.sgen['q_mvar'] = zero_action
        pp.runpp(net, algorithm='bfsw', init='dc', numba=True)
        v[t] = net.res_bus.vm_pu.to_numpy()

    np.save('data/nonlinear_voltage_baseline.npy', v)


if __name__ == '__main__':
    main()
