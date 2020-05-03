import os
import yaml
import io
def load_hyperparams(env_id, algo):
    hypar_dir = os.path.dirname(__file__)
    hypar_fn = os.path.join(hypar_dir, '{}.yml'.format(algo))
    assert os.path.exists(hypar_fn), '{} not found'.format(hypar_fn)
    with io.open(hypar_fn, 'r') as src:
        hyperparams_dict = yaml.safe_load(src)
        if env_id in hyperparams_dict:
            hyperparams = hyperparams_dict[env_id]
        elif 'NoFrameskip' in env_id:
            hyperparams = hyperparams_dict['atari']
        else:
            raise ValueError("Hyperparameters not found for {}-{}".format(algo, env_id))
    for k in hyperparams.keys():
        if isinstance(hyperparams[k], str) and hyperparams[k].startswith('lin'):
            coff = float(hyperparams[k].split('_')[1])
            hyperparams[k] = lambda x : coff * x

    return hyperparams