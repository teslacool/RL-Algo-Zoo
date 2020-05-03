import numpy as np
import random
import torch
def set_global_seeds(i):
    try:
        import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    myseed = i  + 1000 * rank if i is not None else None
    torch.manual_seed(myseed)
    np.random.seed(myseed)
    random.seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(myseed)