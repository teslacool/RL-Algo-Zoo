import sys
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import torch
import numpy as np
import difflib
import os
import rlalgo
from rlalgo import options
from hyperparams import load_hyperparams
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from common import logger
from common.buildenv import build_env

def check_valid_env(env_id):
    registered_envs = set(gym.envs.registry.env_specs.keys())
    if env_id not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError('{} not found in gym registry, you maybe meant {}?'.format(env_id, closest_match))

def check_valid_algo(algo):
    registered_algo = set(rlalgo.VALID_ALGO.keys())
    if algo not in registered_algo:
        try:
            closest_match = difflib.get_close_matches(algo, registered_algo, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError('{} not found in algo registry, you maybe meant {}?'.format(algo, closest_match))




def configure_logger(log_path, **kwargs):

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure(log_path, **kwargs)
    else:
        logger.configure(log_path)


def main():
    args = options.get_training_parser()
    env_id = args.env
    check_valid_env(env_id)
    algo = args.algo
    check_valid_algo(algo)

    logpath = os.path.join(args.log_path, args.exp_name)
    if args.rm_prev_log and osp.exists(logpath):
        import shutil
        shutil.rmtree(logpath)
    configure_logger(logpath)

    env_fn = lambda norm : build_env(args, norm=norm)
    args.env_fn = env_fn
    algo = rlalgo.build_algo(args)

    if not args.play:
        algo.train()
    else:
        algo.play()

    algo.train_env.close()

if __name__ == '__main__':
    main()