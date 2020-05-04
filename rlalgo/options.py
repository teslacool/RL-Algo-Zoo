import argparse
from rlalgo import add_args
from hyperparams import load_hyperparams
import torch
from torch import nn
from common.buildenv import get_env_type

def get_training_parser():
    parser, args = get_common_parser('training')
    algo = args.algo
    add_args(algo, parser)
    args, unknowargs = parser.parse_known_args()
    parser_extra(args, unknowargs)
    envtype, envid = get_env_type(args)
    hyparams = load_hyperparams(envid, algo, envtype=envtype)
    hyparams.update(vars(args))
    vars(args).update(hyparams)
    return args


def parser_extra(args, unknowargs):
    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    extra_args = {}
    for i, arg in enumerate(unknowargs):
        assert i > 0 or '--' in arg
        if '--' in arg:
            arg_key = arg.lstrip('-')
            extra_args[arg_key] = []
        else:
            extra_args[arg_key].append(process(arg))

    for key in extra_args.keys():
        if len(extra_args[key]) == 0:
            extra_args[key] = True
        else:
            assert len(extra_args[key]) == 1
            extra_args[key] = extra_args[key][0]

    def set_args(args, key, value):
        if ':' in key:
            pre, suf = key.split(':', 1)
            if getattr(args, pre, None) is None:
                setattr(args, pre, {suf: value})
            else:
                getattr(args, pre)[suf] = value
        else:
            setattr(args, key, value)

    for k, v in extra_args.items():
        set_args(args, k, v)

def get_common_parser(desc):
    parser = argparse.ArgumentParser(desc)
    common_group = parser.add_argument_group('common configuration',
                                             argument_default=argparse.SUPPRESS)
    common_group.add_argument('--env', help='environment ID', type=str, default='PongNoFrameskip-v4')
    common_group.add_argument('--env_type', help='type of environment, used when the environment type cannot be automatically determined',
                              type=str, default=None)
    common_group.add_argument('--algo', help='Algorithm', type=str, default='ppo')
    common_group.add_argument('--log-freq', help='Override log interval (default: -1, no change)', default=100,
                        type=int)
    common_group.add_argument('--eval-freq', help='Evaluate the agent every n steps (if negative, no evaluation)',
                        default=-1, type=int)
    common_group.add_argument('--save-freq', help='Save the model every n steps (if negative, no checkpoint)',
                        default=-1, type=int)
    common_group.add_argument('--seed', help='RNG seed', type=int, default=0)
    common_group.add_argument('--n_timesteps', type=float),
    common_group.add_argument('--num_env',
                        help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco',
                        type=int)
    common_group.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    common_group.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    common_group.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
    common_group.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
    common_group.add_argument('--log_path', help='Directory to save learning curve data.', default='logs', type=str)
    common_group.add_argument('--rm_prev_log', default=False, action='store_true')
    common_group.add_argument('--play', default=False, action='store_true')
    common_group.add_argument('--exp_name', default='debug', type=str)
    common_group.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    args, _ = parser.parse_known_args()
    return parser, args
