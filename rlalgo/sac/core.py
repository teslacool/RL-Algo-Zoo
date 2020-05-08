import numpy as np
import scipy.signal
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from rlalgo.models import Mlp, weights_init
from torch.distributions import Normal
from rlalgo.models import tensor2np
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def action4env(action):
    if isinstance(action, torch.Tensor):
        return tensor2np(action)
    return action


LOG_STD_MAX = 2
LOG_STD_MIN = -20
class SacActor(nn.Module):

    def __init__(self, latent_module, ac_space, indim=256):
        super().__init__()
        assert isinstance(ac_space, gym.spaces.Box)
        self.latent = latent_module
        self.ncat = ac_space.shape[0]
        self.mean = nn.Linear(indim, self.ncat)
        self.logstd = nn.Linear(indim, self.ncat,)
        self.reset_parameter()
        self.action_scale = ac_space.high[0]

    def reset_parameter(self):
        weights_init(self.mean, gain=0.01)
        weights_init(self.logstd, gain=1)

    def _distribution(self, obs):
        latent = self.latent(obs)
        mean = self.mean(latent)
        logstd = self.logstd(latent)
        logstd = torch.clamp(logstd, min=LOG_STD_MIN, max=LOG_STD_MAX)
        std = torch.exp(logstd)
        return Normal(mean, std)

    def forward(self, x):
        return self._distribution(x)

    def step(self, x, with_logprob=True, deterministic=False):
        dist = self.forward(x)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        if with_logprob:
            logp = dist.log_prob(action).sum(axis=-1)
            logp -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=1)
        else:
            logp = None

        action = torch.tanh(action)
        action = action *  self.action_scale

        return action, logp



class SacCritic(nn.Module):

    def __init__(self, latent_module, indim=256):
        super().__init__()
        self.latent = latent_module
        self.matching = nn.Linear(indim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self.matching, gain=1.)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        latent = self.latent(x)
        return self.matching(latent).squeeze(-1)


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, env=None, **ac_kwargs):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        latent = Mlp(indim=observation_space.shape[0],
                     num_hidden=ac_kwargs.get('num_hidden', 256),
                     acti=ac_kwargs.get('activation', nn.ReLU))
        self.actor = SacActor(latent, action_space,)
        latent = Mlp(indim=observation_space.shape[0] + action_space.shape[0],
                     num_hidden=ac_kwargs.get('num_hidden', 256),
                     acti=ac_kwargs.get('activation', nn.ReLU))
        self.q1 = SacCritic(latent,)
        latent = Mlp(indim=observation_space.shape[0] + action_space.shape[0],
                     num_hidden=ac_kwargs.get('num_hidden', 256),
                     acti=ac_kwargs.get('activation', nn.ReLU))
        self.q2 = SacCritic(latent,)

    def act(self, obs, deterministic=False):
        obs = obs.to(torch.float32)
        with torch.no_grad():
            return self.actor.step(obs, with_logprob=False, deterministic=deterministic)[0]

    def step(self, obs):
        obs = obs.to(torch.float32)
        return self.actor.step(obs)

