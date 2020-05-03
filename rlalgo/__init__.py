import importlib
import os
import os.path as osp
VALID_ALGO = {}
import inspect

def build_algo(args,):
    algo = getattr(args, 'algo', None)
    assert algo in VALID_ALGO
    algo = VALID_ALGO[algo]
    init_args = {}
    for p in inspect.signature(algo).parameters.values():
        if (
            p.kind == p.POSITIONAL_ONLY
            or p.kind == p.VAR_POSITIONAL
            or p.kind == p.VAR_KEYWORD
        ):
            # we haven't implemented inference for these argument types,
            # but PRs welcome :)
            raise NotImplementedError('{} not supported'.format(p.kind))

        assert p.kind in {p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY}

        if hasattr(args, p.name):
            init_args[p.name] = getattr(args, p.name)
        elif p.default != p.empty:
            pass  # we'll use the default value
        else:
            raise NotImplementedError(
                'Unable to infer Criterion arguments, please implement '
                '{}.build_criterion'.format(algo.__name__)
            )
    return algo(**init_args)



def add_args(algo, parser):
    VALID_ALGO[algo].add_args(parser)

def register_algo(name):

    def register_algo_cls(cls):
        if name in VALID_ALGO:
            raise ValueError('Cannot register duplicate rlalgo ({})'.format(name))
        # if not issubclass(cls, RLModel):
        #     raise ValueError('Model ({}: {}) must extend BaseRLModel'.format(name, cls.__name__))
        VALID_ALGO[name] = cls
        return cls

    return register_algo_cls

algo_dir = osp.dirname(__file__)
for algo_name in os.listdir(algo_dir):
    path = osp.join(algo_dir, algo_name)
    if osp.isdir(path):
        algo_path = osp.join(path, '{}.py'.format(algo_name))
        if osp.exists(algo_path):
            importlib.import_module('rlalgo.{}.{}'.format(algo_name, algo_name))




