from copy import deepcopy
import itertools
import numpy as np
import torch
import os, math
from torch.optim import Adam
import gym
import time
from . import core
from .core import action4env
from common import logger
from rlalgo import register_algo
from rlalgo.models import preparemodel, np2tentor, tensor2np
import argparse
from common.vec_env import VecNormalize


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size, num_env=1):
        self.obs_buf = np2tentor(np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32))
        self.obs2_buf = np2tentor(np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32))
        self.act_buf = np2tentor(np.zeros(core.combined_shape(size, act_dim), dtype=np.float32))
        self.rew_buf = np2tentor(np.zeros(size, dtype=np.float32))
        self.done_buf = np2tentor(np.zeros(size, dtype=np.bool))
        self.ptr, self.size, self.max_size = 0, 0, size
        self.num_env = num_env
        assert self.max_size % self.num_env == 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr: self.ptr+self.num_env] = np2tentor(obs)
        self.obs2_buf[self.ptr: self.ptr+self.num_env] = np2tentor(next_obs)
        self.act_buf[self.ptr: self.ptr+self.num_env] = np2tentor(act)
        self.rew_buf[self.ptr: self.ptr+self.num_env] = np2tentor(rew)
        self.done_buf[self.ptr: self.ptr+self.num_env] = np2tentor(done)
        self.ptr = (self.ptr+self.num_env) % self.max_size
        self.size = min(self.size+self.num_env, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])



def constfn(val):
    def f(_):
        return val
    return f


@register_algo('sac')
class sac(object):

    def __init__(self, env_fn, actor_critic='mlp', ac_kwargs=dict(),
                 nsteps=2048, n_timesteps=1e6, gamma=0.99, replay_size=int(1e6),
                 polyak=0.995, lr=3e-4, batch_size=256, start_steps=10000,
                 update_after=1000, num_test_episodes=10, max_ep_len=1000,
                 log_freq=10, load=False, update_every=48, alpha=0.2,
                 env_norm=False, num_env=1):
        self.batch_size = batch_size
        self.start_steps =start_steps
        self.update_after = update_after
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.update_every = update_every
        self.log_freq = log_freq
        self.nsteps = nsteps
        self.n_timesteps = n_timesteps
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        # mujoco num_env=1
        self.train_env = env_fn({'norm': env_norm, 'numenv': num_env})
        self.nenv = self.train_env.num_envs
        assert self.update_every % self.nenv == 0
        self.test_env =  env_fn({'norm': env_norm, 'numenv': num_test_episodes})
        self.ac = core.MLPActorCritic(self.train_env.observation_space, self.train_env.action_space,
                                      env=self.train_env, **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        if isinstance(self.train_env, VecNormalize):
            self.test_env.ob_rms = self.train_env.ob_rms
            self.test_env.ret_rms = self.test_env.ret_rms
        self.buffer = ReplayBuffer(obs_dim=self.train_env.observation_space.shape[0],
                                   act_dim=self.train_env.action_space.shape[0],
                                   size=replay_size,
                                   num_env=num_env)
        var_counts = tuple(core.count_vars(module) for module in [self.ac.actor, self.ac.q1, self.ac.q2])
        logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)


        if isinstance(lr, float): lr=constfn(lr)
        else: assert callable(lr)
        lr_start = lr(1.)

        self._epoch = 0
        self._t = 0
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=lr_start)
        self.total_epoch = math.ceil(n_timesteps / nsteps)
        self.lr_lambda = lr_lambda = lambda epoch: lr(max(1e-4, 1 - epoch / self.total_epoch)) / lr_start
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        preparemodel(self.ac)
        preparemodel(self.ac_targ)



    def save_model(self, modelfn=None):
        modelfn = modelfn if modelfn else 'checkpoint.pt'
        modelpath = os.path.join(logger.get_dir(), 'models', modelfn)
        os.makedirs(os.path.join(logger.get_dir(), 'models'), exist_ok=True)
        state_dict = {
            'epoch': self._epoch + 1,
            'state_dict': self.ac.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
        }
        torch.save(state_dict, modelpath)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(logger.get_dir(), 'models', 'checkpoint.pt')
        state_dict = torch.load(model_path)
        self._epoch = state_dict['epoch']
        self.ac.load_state_dict(state_dict['state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])


    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        algo_group = parser.add_argument_group('algo configuration', argument_default=argparse.SUPPRESS)
        algo_group.add_argument('--actor_critic', type=str, )
        algo_group.add_argument('--lr', type=float, )
        algo_group.add_argument('--load', action='store_true')
        algo_group.add_argument('--env_norm', action='store_true')
        return parser

    def train(self):

        o = self.train_env.reset()
        first_tstart = time.perf_counter()
        for _epoch in range(self._epoch, self.total_epoch):
            tstart = time.perf_counter()
            for _t in range(0, self.nsteps, self.nenv):

                if self._t > self.start_steps:
                    a = self.ac.act(np2tentor(o))
                    a = action4env(a)
                else:
                    a = np.concatenate([self.train_env.action_space.sample().reshape(1, -1)
                                        for _ in range(self.nenv)], axis=0)
                o2, r, d, infos = self.train_env.step(a)
                self.buffer.store(o, a, r, o2, d)
                o = o2
                for info in infos:
                    maybeepinfo = info.get('episode')
                    if maybeepinfo:
                        logger.logkv_mean('eprewtrain', maybeepinfo['r'])
                        logger.logkv_mean('eplentrain', maybeepinfo['l'])
                self._t += self.nenv
                if self._t >= self.update_after and self._t % self.update_every == 0:
                    self.update()
                if self._t > self.n_timesteps:
                    break

            fps = int(_t / (time.perf_counter() - tstart))

            if (_epoch % self.log_freq ==0 or _epoch == self.total_epoch - 1):
                self.test_agent()
                logger.logkv('epoch', _epoch)
                logger.logkv('lr', self.optimizer.param_groups[0]['lr'])
                logger.logkv('timesteps', self._t)
                logger.logkv('fps', fps)
                logger.logkv('time_elapsed', time.perf_counter() - first_tstart)
                logger.dump_tabular()
                self._epoch = _epoch
                # self.save_model()

    def update(self):
        for _ in range(self.update_every):
            self.optimizer.zero_grad()

            data = self.buffer.sample_batch(self.batch_size)
            o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

            # compute loss q
            q1 = self.ac.q1(o, a)
            q2 = self.ac.q2(o, a)

            with torch.no_grad():
                a2, logpa2 = self.ac.step(o2)
                q1_targ = self.ac_targ.q1(o2, a2)
                q2_targ = self.ac_targ.q2(o2, a2)
                q_targ = torch.min(q1_targ, q2_targ)
                backup = r + (self.gamma * (q_targ - self.alpha * logpa2)).masked_fill(d, 0.)
            loss_q1 = torch.mean(torch.square(q1 - backup))
            loss_q2 = torch.mean(torch.square(q2 - backup))
            loss_q = loss_q1 + loss_q2
            loss_q.backward()
            # compute loss pi
            for p in self.ac.q1.parameters():
                p.requires_grad = False
            for p in self.ac.q2.parameters():
                p.requires_grad = False
            pi, logp_pi = self.ac.step(o)
            q1_pi = self.ac.q1(o, pi)
            q2_pi = self.ac.q2(o, pi)
            q_pi = torch.min(q1_pi, q2_pi)
            loss_pi = torch.mean((self.alpha * logp_pi - q_pi))
            loss_pi.backward()
            self.optimizer.step()
            for p in self.ac.q1.parameters():
                p.requires_grad = True
            for p in self.ac.q2.parameters():
                p.requires_grad = True
            logger.logkv_mean('loss_q', loss_q.item())
            logger.logkv_mean('loss_pi', loss_pi.item())
            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

    def test_agent(self):
        for j in range(self.num_test_episodes):
            o = self.test_env.reset()
            i = 0
            while True:
                # Take deterministic actions at test time
                a = self.ac.act(np2tentor(o), deterministic=True)
                o, r, d, infos = self.test_env.step(tensor2np(a))
                for info in infos:
                    maybeepinfo = info.get('episode')
                    if maybeepinfo:
                        logger.logkv_mean('eprewmean', maybeepinfo['r'])
                        logger.logkv_mean('eplenmean', maybeepinfo['l'])
                        i += 1
                        if i == 10:
                            return









