import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import gym
import math
import numpy as np

def np2tentor(t):
    if torch.cuda.is_available():
        return torch.from_numpy(t).cuda()
    else:
        return torch.from_numpy(t)

def tensor2np(t):
    if torch.cuda.is_available():
        return t.detach().cpu().numpy()
    else:
        return t.detach().numpy()

def preparemodel(m):
    if torch.cuda.is_available():
        m.cuda()

def baselines_orthogonal_(tensor, gain=1):

    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor.numel() // rows
    flattened = np.random.normal(0., 1., (cols, rows))
    u, _, v = np.linalg.svd(flattened, full_matrices=False)
    q = u if u.shape == (cols, rows) else v
    assert q.shape == (cols, rows)
    q = q.transpose()
    assert q.shape[0] == tensor.shape[0]
    q = q.reshape(tensor.shape)


    with torch.no_grad():
        tensor.copy_(torch.from_numpy(q))
        tensor.mul_(gain)
    return tensor


def weights_init(m, gain=float(math.sqrt(2))):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        baselines_orthogonal_(m.weight.data, gain=gain)
    if classname.find('Linear') != -1:
        baselines_orthogonal_(m.weight.data, gain=gain)

class NatureCnn(nn.Module):

    def __init__(self, channel_last=True, indim=84):
        super(NatureCnn, self).__init__()
        self.channel_last = channel_last
        self.activ = nn.ReLU()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        getCnnOut = lambda w, k, s: int((w - k) / s) + 1
        convout = getCnnOut(getCnnOut(getCnnOut(indim, 8, 4),
                                      4, 2), 3, 1)
        self.fc = nn.Linear(convout * convout * 64, 512)
        # TODO: init param like baselines
        self.reset_parameter()

    def reset_parameter(self):
        init_f = lambda m: weights_init(m)
        self.conv.apply(init_f)
        self.fc.apply(init_f)

    def forward(self, x):
        x = x / 255.
        if self.channel_last:
            # permute from NHWC to NCHW
            x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.activ(x)

class Mlp(nn.Module):

    def __init__(self, indim, num_layers=2, num_hidden=64, acti=torch.nn.Tanh, layernorm=False):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        if layernorm:
            self.lns = nn.ModuleList()
        self.layernorm = layernorm
        self.activations = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(indim if i == 0 else num_hidden, num_hidden))
            if layernorm:
                self.lns.append(torch.nn.LayerNorm(num_hidden))
            self.activations.append(acti())
        self.reset_parameter()

    def reset_parameter(self):
        init_f = lambda m: weights_init(m)
        self.layers.apply(init_f)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i in range(self.num_layers):
            x = self.layers[i](x)
            if self.layernorm:
                x = self.lns[i](x)
            x = self.activations[i](x)
        return x




class CategoricalActor(nn.Module):

    def __init__(self, latent_module, ac_space, indim=512):
        super().__init__()
        assert isinstance(ac_space, gym.spaces.Discrete)
        self.latent = latent_module
        self.ncat = ac_space.n
        if indim == self.ncat:
            self.matching = None
        else:
            self.matching = nn.Linear(indim, self.ncat)
        self.reset_parameters()

    def reset_parameters(self):
        if self.matching is not None:
            weights_init(self.matching, gain=0.01)

    def _distribution(self, obs):
        latent = self.latent(obs)
        if self.matching:
            logits = self.matching(latent)
        else:
            logits = latent

        return Categorical(logits=logits), latent

    def forward(self, x):
        return self._distribution(x)


class NormalActor(nn.Module):

    def __init__(self, latent_module, ac_space, indim=64):
        super().__init__()
        assert isinstance(ac_space, gym.spaces.Box)
        self.latent = latent_module
        self.nact = ac_space.shape[0]
        if indim == self.nact:
            self.mean = None
        else:
            self.mean = nn.Linear(indim, self.nact)
        self.logstd = nn.Parameter(torch.zeros(1, self.nact))
        self.reset_parameter()

    def reset_parameter(self):
        if self.mean is not None:
            weights_init(self.mean, gain=0.01)

    def _distribution(self, obs):
        latent = self.latent(obs)
        if self.mean:
            mean = self.mean(latent)
        else:
            mean = latent
        logstd = mean * 0 + self.logstd
        logstd = torch.exp(logstd)

        return Normal(mean, logstd), latent

    def forward(self, x):
        return self._distribution(x)



class CategoricalCritic(nn.Module):

    def __init__(self, latent_module, ac_space, indim=512, estimate_q=False):
        super().__init__()
        assert isinstance(ac_space, gym.spaces.Discrete)
        self.latent = latent_module
        self.ncat = ac_space.n
        if estimate_q:
            self.matching = nn.Linear(indim, self.ncat)
        else:
            self.matching = nn.Linear(indim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self.matching, gain=1.)

    def forward(self, obs, latent=None, isObs=True):
        assert latent is not None or isObs
        if isObs:
            latent = self.latent(obs)
            return self.matching(latent).squeeze(-1)
        else:
            return self.matching(latent).squeeze(-1)

class NormalCritic(nn.Module):

    def __init__(self, latent_module, ac_space, indim=64, estimate_q=False):
        super().__init__()
        assert isinstance(ac_space, gym.spaces.Box)
        self.latent = latent_module
        self.nact = ac_space.shape[0]
        if estimate_q:
            raise NotImplementedError
        else:
            self.matching = nn.Linear(indim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self.matching, gain=1.)

    def forward(self, obs, ):
        latent = self.latent(obs)
        return self.matching(latent).squeeze(-1)







