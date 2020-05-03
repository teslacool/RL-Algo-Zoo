import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from rlalgo.models import NatureCnn, CategoricalActor, CategoricalCritic, \
    Mlp, NormalActor, NormalCritic

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

class CNNActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, **ac_kwargs):
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

    def __init__(self, observation_space, action_space, **ac_kwargs):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        latent = Mlp(indim=self.observation_space.shape[0])
        self.actor = NormalActor(latent, action_space)
        latent = Mlp(indim=self.observation_space.shape[0])
        self.critic = NormalCritic(latent, action_space)

    def forward(self, obs):
        obs = obs.to(torch.float32)
        dist, latent = self.actor(obs)
        v = self.critic(obs)
        return dist, v

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


