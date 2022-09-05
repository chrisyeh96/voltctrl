import numpy as np
from numpy import linalg as LA
import gym
import os
import random
import sys
from gym import spaces
from gym.utils import seeding
import copy

from scipy.io import loadmat
import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import cvxpy as cp



class VoltageCtrl_nonlinear(gym.Env):
    def __init__(self, pp_net, vmax, vmin, v0, injection_bus, obs_dim=55, action_dim=55):

        self.network =  pp_net
        self.injection_bus = injection_bus
        self.agentnum = len(injection_bus)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.v0 = v0
        self.vmax = vmax
        self.vmin = vmin

        self.load0_p = np.copy(self.network.load['p_mw'])
        self.load0_q = np.copy(self.network.load['q_mvar'])

        self.gen0_p = np.copy(self.network.sgen['p_mw'])
        self.gen0_q = np.copy(self.network.sgen['q_mvar'])

        self.state = np.ones(self.agentnum, )

    def step(self, action):
        "State transition dynamics: it takes in the reactive power setpoint"
        "then compute the voltage magnitude at each node via solving power flow"

        done = False

        reward = float(-100*LA.norm(self.state-1.0)**2)

        # state-transition dynamics
        for i in range(len(self.injection_bus)):
            self.network.sgen.at[i, 'q_mvar'] = action[i]

        pp.runpp(self.network, algorithm='bfsw', init = 'dc')

        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()

        return self.state, reward, done


    def step_load_solar(self, action, load_p, load_q, gen_p, gen_q):
        "State transition dynamics: it takes in the reactive power setpoint, load_p and load_q"
        "and gen_p & gen_q to compute the voltage magnitude at each node via solving power flow"

        done = False

        reward = float(-100*LA.norm(self.state-1.0)**2)

        # state-transition dynamics
        self.network.load['p_mw'] = load_p
        self.network.load['q_mvar'] = load_q
        self.network.sgen['p_mw'] = gen_p
        self.network.sgen['q_mvar'] = gen_q

        for i in range(len(self.injection_bus)):
            self.network.sgen.at[i, 'q_mvar'] += action[i]

        pp.runpp(self.network, algorithm='bfsw', init = 'dc')

        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()

        return self.state, reward, done

    def reset(self, seed=1):
        np.random.seed(seed)
        self.network.sgen['p_mw'] = 0.0
        self.network.sgen['q_mvar'] = 0.0
        self.network.load['p_mw'] = 0.0
        self.network.load['q_mvar'] = 0.0


        pp.runpp(self.network, algorithm='bfsw')
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        return self.state