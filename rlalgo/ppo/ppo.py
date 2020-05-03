from rlalgo import register_algo
import argparse
from . import core
import torch
import numpy as np
from common.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from common.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from common import logger
from rlalgo.models import preparemodel, np2tentor, tensor2np
import time
from collections import deque

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def constfn(val):
    def f(_):
        return val
    return f

@register_algo('ppo')
class ppo(object):

    def __init__(self, env_fn, actor_critic='cnn', ac_kwargs=dict(),
        nsteps=2048, n_timesteps=1e6, gamma=0.99, clip_ratio=0.2, lr=3e-4,
        vf_coef=0.5, ent_coef=0.0, lam=0.95, max_ep_len=1000, max_grad_norm=0.5,
        save_freq=10, eval_freq=10, log_freq=1, nminibatches=4, noptepochs=4,):

        self.log_freq = log_freq
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.lam = lam
        self.max_ep_len = max_ep_len
        self.max_grad_norm = max_grad_norm
        self.noptepochs = noptepochs
        self.gamma = gamma
        self.nsteps = nsteps
        setup_pytorch_for_mpi()

        self.train_env = env_fn(False)
        self.obs = self.train_env.reset()
        if actor_critic == 'cnn':
            actor_critic = core.CNNActorCritic
        else:
            actor_critic = core.MlpActorCritic
        self.ac = actor_critic(self.train_env.observation_space, self.train_env.action_space, **ac_kwargs)
        preparemodel(self.ac)

        sync_params(self.ac)
        var_counts = tuple(core.count_vars(module) for module in [self.ac.actor, self.ac.critic])
        logger.log('\nNumber of parameters: \t actor: %d, \t critic: %d\n'%var_counts)

        if isinstance(lr, float): lr = constfn(lr)
        else: assert callable(lr)
        if isinstance(clip_ratio, float): clip_ratio = constfn(clip_ratio)
        else: assert callable(clip_ratio)
        self.clip_ratio = clip_ratio

        self.nenv = nenv = self.train_env.num_envs
        self.dones = np.array([False for _ in range(nenv)])
        self.ob_space = self.train_env.observation_space
        self.ac_space = self.train_env.action_space
        self.is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

        self.epinfobuf = deque(maxlen=100)
        self.nbatch = nbatch = nenv * nsteps
        assert nbatch % nminibatches == 0
        self.nbatch_train = nbatch // nminibatches

        self.total_epoch = int(n_timesteps // nbatch)

        lr_start = lr(1.)
        self.lr_lambda = lr_lambda = lambda epoch:  lr(max(1e-4, 1 - epoch / self.total_epoch )) / lr_start
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=lr_start)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.states = None

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        algo_group = parser.add_argument_group('algo configuration', argument_default=argparse.SUPPRESS)
        algo_group.add_argument('--actor_critic', type=str, )
        return parser



    def train(self):
        first_tstart = time.perf_counter()
        for _epoch in range(self.total_epoch):
            tstart = time.perf_counter()
            frac = 1. - _epoch * 1. / self.total_epoch
            clip_ratio_now = self.clip_ratio(frac)
            if _epoch % self.log_freq == 0 and self.is_mpi_root:
                logger.log('Stepping environment...')

            # collect data
            obs, returns, masks, actions, values, neglogpacs, states, epinfos = self.collect()
            # if eval_env is not None:
            #     eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, eval_epinfos = eval_runner.run()  # pylint: disable=E0632

            if _epoch % self.log_freq == 0 and self.is_mpi_root:
                logger.log('done')

            self.epinfobuf.extend(epinfos)
            # if eval_env is not None:
            #     eval_epinfobuf.extend(eval_epinfos)
            self.update(obs, returns, masks, actions, values, neglogpacs, clip_ratio_now, states)
            self.lr_scheduler.step()
            fps = int(self.nbatch / (time.perf_counter() - tstart))
            if _epoch % self.log_freq == 0 and self.is_mpi_root:
                logger.logkv('epoch', _epoch)
                logger.logkv('lr', self.optimizer.param_groups[0]['lr'])
                logger.logkv('timesteps', (_epoch + 1) * self.nbatch)
                logger.logkv('fps', fps)
                logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in self.epinfobuf]))
                logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in self.epinfobuf]))
                logger.logkv('time_elapsed', time.perf_counter() - first_tstart)
                logger.dump_tabular()

    def update(self, obs, returns, masks, actions, values, neglogpacs, clip_ratio, states):
        if states is None:
            for _ in range(self.noptepochs):
                permutation = np.random.permutation(self.nbatch)
                for start in range(0, self.nbatch, self.nbatch_train):
                    end = start + self.nbatch_train
                    mbinds = permutation[start:end]
                    mbinds = np2tentor(mbinds)
                    obs_mnb, returns_mnb, masks_mnb, actions_mnb, values_mnb, neglogpacs_mnb = \
                        (arr.index_select(0, mbinds) for arr in (obs, returns, masks, actions, values, neglogpacs))
                    advs_mnb = returns_mnb - values_mnb
                    advs_mnb = (advs_mnb - advs_mnb.mean()) / (advs_mnb.std() + 1e-8)
                    dist, v_now = self.ac(obs_mnb)
                    neglogpacs_now = self.ac.neglogprob(dist, actions_mnb)
                    entropy = dist.entropy().mean()

                    clip_v = values_mnb + torch.clamp(v_now - values_mnb, -clip_ratio, clip_ratio)
                    vf_loss1 = torch.square(v_now - returns_mnb)
                    vf_loss2 = torch.square(clip_v - returns_mnb)
                    vf_loss = 0.5 * torch.mean(torch.max(vf_loss1, vf_loss2))

                    ratio = torch.exp(neglogpacs_mnb - neglogpacs_now)
                    pg_losses1 = - advs_mnb * ratio
                    pg_losses2 = - advs_mnb * torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
                    pg_losses = torch.mean(torch.max(pg_losses1, pg_losses2))
                    with torch.no_grad():
                        approxkl = torch.mean(neglogpacs_now - neglogpacs_mnb)
                        clipfrac = torch.mean((torch.abs(ratio - 1.) > clip_ratio).float())

                    loss = pg_losses - entropy * self.ent_coef + vf_loss * self.vf_coef
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    logger.logkv_mean('total_loss', loss.item())
                    logger.logkv_mean('pg_loss', loss.item())
                    logger.logkv_mean('entropy', entropy.item())
                    logger.logkv_mean('vf_loss', vf_loss.item())
                    logger.logkv_mean('approxkl', approxkl.item())
                    logger.logkv_mean('clipfrac', clipfrac.item())

    def collect(self):
        with torch.no_grad():
            mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
            mb_states = self.states
            epinfos = []
            for _ in range(self.nsteps):
                obs = np2tentor(self.obs)
                actions, values, self.states, neglogpacs = self.ac.step(obs)
                mb_obs.append(obs.detach())
                mb_actions.append(actions.detach())
                mb_values.append(values.detach())
                mb_neglogpacs.append(neglogpacs.detach())
                mb_dones.append(np2tentor(self.dones))
                actions = tensor2np(actions)
                self.obs, rewards, self.dones, infos = self.train_env.step(actions)
                for info in infos:
                    maybeepinfo = info.get('episode')
                    if maybeepinfo: epinfos.append(maybeepinfo)
                mb_rewards.append(np2tentor(rewards))
            mb_obs = torch.stack(mb_obs, dim=1)
            mb_rewards = torch.stack(mb_rewards, dim=1)
            mb_actions = torch.stack(mb_actions, dim=1)
            mb_values = torch.stack(mb_values, dim=1)
            mb_neglogpacs = torch.stack(mb_neglogpacs, dim=1)
            mb_dones = torch.stack(mb_dones, dim=1)
            last_values = self.ac.value(np2tentor(self.obs)).detach()

            # discount/bootstrap off value fn
            mb_advs = mb_rewards.new_zeros(mb_rewards.size())
            lastgaelam = mb_rewards.new_zeros(mb_rewards.size(0))
            for t in reversed(range(self.nsteps)):
                if t == self.nsteps - 1:
                    nextterminal = np2tentor(self.dones)
                    nextvalues = last_values
                else:
                    nextterminal = mb_dones[:, t + 1]
                    nextvalues = mb_values[:, t + 1]
                delta = mb_rewards[:, t] +  (self.gamma * nextvalues).masked_fill_(nextterminal, 0) - mb_values[:, t]
                mb_advs[:, t] = lastgaelam = delta + (self.gamma * self.lam * lastgaelam).masked_fill_(nextterminal, 0)
            mb_returns = mb_advs + mb_values

            return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
                mb_states, epinfos)



def sf01(t):
    s = t.shape
    return t.view(s[0]*s[1], *s[2:])

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)






