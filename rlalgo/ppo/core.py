import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from rlalgo.models import NatureCnn, CategoricalActor, CategoricalCritic, \
    Mlp, NormalActor, NormalCritic

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class CNNActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, env=None, **ac_kwargs):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        latent = NatureCnn(indim=self.observation_space.shape[0])
        self.actor = CategoricalActor(latent, action_space)
        self.critic = CategoricalCritic(latent, action_space)

    def forward(self, obs):
        dist, latent = self.actor(obs)
        v = self.critic(obs, latent=latent, isObs=False)
        return dist, v

    def step(self, obs):

        dist, v = self.forward(obs)
        action = dist.sample()
        neglogpacs = - dist.log_prob(action)
        return action, v, None, neglogpacs

    def value(self, obs):

        v = self.critic(obs)
        return v

    def neglogprob(self, dist, action):
        return - dist.log_prob(action)


class MlpActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, env=None, **ac_kwargs ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        latent = Mlp(indim=self.observation_space.shape[0])
        self.actor = NormalActor(latent, action_space)
        latent = Mlp(indim=self.observation_space.shape[0])
        self.critic = NormalCritic(latent, action_space)
        self.set_env_rms(env)

    def set_env_rms(self, env):
        from common.vec_env.vec_normalize import VecNormalize
        from common.running_mean_std import TorchRunningMeanStd
        self.env_obs_rms = None
        self.env_ret_rms = None
        if env is not None and isinstance(env, VecNormalize) \
                and  isinstance(env.ob_rms, TorchRunningMeanStd):
            self.env_obs_rms = env.ob_rms
            self.env_ret_rms = env.ret_rms

    def forward(self, obs):
        obs = obs.to(torch.float32)
        dist, latent = self.actor(obs)
        v = self.critic(obs)
        return dist, v

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict=state_dict, strict=strict)
        if self.env_obs_rms:
            self.env_ret_rms._set_para()
            self.env_obs_rms._set_para()

    def step(self, obs):
        dist, v = self.forward(obs)
        action = dist.sample()
        neglogpacs = - torch.sum(dist.log_prob(action),dim=1)
        return action, v, None, neglogpacs

    def value(self, obs):
        obs = obs.to(torch.float32)
        v = self.critic(obs)
        return v

    def neglogprob(self, dist, action):
        return - torch.sum(dist.log_prob(action), dim=1)



