from dataclasses import dataclass

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from typing import Callable

import numpy as np

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

@dataclass
class ToyModelConfig:
    input_size: int = 6
    hidden_size: int = 2
    feat_set_size: int = 1
    sparsity: Callable = lambda x: 0.1*t.ones_like(x)
    importance: Callable = lambda x: t.ones_like(x)

@dataclass
class ToyModelTrainConfig:
    batches: int = 1000
    batch_size: int = 100
    lr: float = 1e-3
    wd: float = 0.01

class TMS(nn.Module):
    """
    "ReLU Output Model" Toy Model of Superposition as described in 
    https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating-setup

    The model consists of a linear input, linked linear output with bias, and ReLU activation.
    """

    def __init__(self, cfg):
        super(TMS, self).__init__()
        self.W = nn.Parameter(t.empty(cfg.input_size, cfg.hidden_size).uniform_(-1/np.sqrt(cfg.input_size), 1/np.sqrt(cfg.input_size)))
        self.b = nn.Parameter(t.zeros(cfg.input_size))

    def forward(self, x):
        h = t.einsum('ih,bi->bh', self.W, x)
        z = t.einsum('ih,bh->bi', self.W, h) + self.b
        return F.relu(z)

    def loss(self, x, y, importance):
        return t.sum(importance[None,:]*(x-y)**2)


class Datagenerator:
    """
    Generates batches of data for the toy model.

    The data is generated as follows:
    - The input is a vector of size input_size.
    - The input is made sparse by setting some of the elements to zero.
    - The sparsity is defined by the sparsity function, which takes a torch Tensor of length set_size as input.
    - The importance of each element is defined by the importance function, which takes a torch Tensor of length input_size as input.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.feat_sets = cfg.input_size//cfg.feat_set_size
        self.S = cfg.sparsity(self.feat_sets)
        self.I = cfg.importance(self.cfg.input_size)
        if cfg.input_size % cfg.feat_set_size != 0:
            raise ValueError('input_size must be divisible by feat_set_size')
        self.set_indices = t.arange(cfg.input_size).reshape(self.feat_sets, cfg.feat_set_size)

    def generate_uncorrelated_batch(self, batch_size):
        """ Makes a batch where each element is kept with probability S."""
        s = t.rand(batch_size, self.cfg.input_size)
        x = t.rand(batch_size, self.cfg.input_size)
        x[s < self.S[None,:]] = 0 #make sparse
        return x

    def generate_correlated_batch(self, batch_size):
        """ Makes a batch where each feature set is kept with probability S."""
        s = t.rand(batch_size, self.feat_sets)
        mask = t.repeat_interleave((s < self.S[None,:]).to(int), 2, dim=-1).to(bool)
        x = t.rand(batch_size, self.cfg.input_size)
        x[mask] = 0 #make sparse
        return x

    def generate_anticorrelated_batch(self, batch_size):
        """ Generated like a correlated batch, but only one element is kept per feature set. """
        x = self.generate_correlated_batch(batch_size)
        mask_int = t.stack([t.cat([t.randperm(cfg.feat_set_size) for _ in range(self.feat_sets)]) for _ in range(batch_size)])
        x *= (mask_int == 0)
        return x

    def __call__(self, batch_size, data_type='uncorrelated'):
        if data_type == 'uncorrelated':
            return self.generate_uncorrelated_batch(batch_size)
        elif data_type == 'correlated':
            return self.generate_correlated_batch(batch_size)
        elif data_type == 'anticorrelated':
            return self.generate_anticorrelated_batch(batch_size)
