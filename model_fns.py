from dataclasses import dataclass

import torch as t
from torch import nn, Tensor
import torch.nn.functional as F
import torch.optim as optim

from typing import Callable
from typing import Optional, Callable, Union, List, Tuple, Dict
from jaxtyping import Float, Int, Bool

import numpy as np

from plot_fns import TMSPlotter
from tqdm.notebook import tqdm
from IPython.display import display, clear_output

device = t.device('cuda' if t.cuda.is_available() else 'cpu')


@dataclass
class ToyModelConfig:
    input_size: int = 6
    hidden_size: int = 2

    #uncorrelated / correlated / anticorrelated cfg options
    feat_set_size: int = 1
    sparsity: Callable = lambda x: 0.1

    #composed features cfg options
    feat_sets: list = None
    correlate_set_magnitudes: bool = False


    #Training parameters
    batches: int = 1000
    batch_size: int = 100
    lr: float = 1e-3
    wd: float = 0.01

    #general data draw options
    importance: Callable = lambda x: t.ones(x)

    def __post_init__(self):
        if self.feat_sets is not None:
            self.input_size = 0
            for i in self.feat_sets:
                self.input_size += i

class TMS(nn.Module):
    def __init__(self, cfg):
        super(TMS, self).__init__()
        self.cfg = cfg
        self.W = nn.Parameter(t.empty(cfg.input_size, cfg.hidden_size).uniform_(-1/np.sqrt(cfg.input_size), 1/np.sqrt(cfg.input_size)))
        self.b = nn.Parameter(t.zeros(cfg.input_size))
        self.h = None
        self.features = None
        self.train_log = None

    def forward(self, x):
        self.h = t.einsum('ih,bi->bh', self.W, x)
        z = t.einsum('ih,bh->bi', self.W, self.h) + self.b
        return F.relu(z)

    def loss(self, x, y, importance):
        return t.mean(importance[None,:]*(x-y)**2)

    def get_batch(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in children classes")
    
    def train(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wd)

        plotter = TMSPlotter()
        progress_bar = tqdm(total = self.cfg.batches)

        self.train_log = {'steps': [], 'losses' : []}

        losses = []
        for i in range(self.cfg.batches):
            optimizer.zero_grad()
            x = self.get_batch(self.cfg.batch_size).to(device)
            y = self(x)
            loss = self.loss(x, y, self.I.to(device))
            loss.backward()
            optimizer.step()
            self.train_log['steps'].append(i+1)
            self.train_log['losses'].append(loss.item())
            progress_bar.set_description(f"step: {i+1}, loss: {loss.item():.3e}")
            progress_bar.update()

            if i % 200 == 0:
                plotter.update(self)
        clear_output()

class UncorrelatedTMS(TMS):

    def __init__(self, cfg):
        super(UncorrelatedTMS, self).__init__(cfg)
        self.S = t.Tensor([cfg.sparsity(i) for i in range(self.cfg.input_size)])
        self.I = cfg.importance(self.cfg.input_size)

    def get_batch(self, batch_size: int = None,
                  identity=False,
                  ) -> Float[Tensor, "batch input_size"]:
        if identity:
            self.features = t.eye(self.cfg.input_size)
        else:
            if batch_size is None:
                batch_size = self.cfg.batch_size

            s = t.rand(batch_size, self.cfg.input_size)
            self.features = t.rand(batch_size, self.cfg.input_size)
            self.features[s < self.S[None,:]] = 0 #make sparse
        return self.features.to(device)


class ComposedFeatureTMS(TMS):

    def __init__(self, cfg):
        super(ComposedFeatureTMS, self).__init__(cfg)
        if self.cfg.feat_sets is None:
            raise ValueError("Must set config feat_sets list for Composed Features Model")
        self.feat_sets = self.cfg.feat_sets
        self.feat_sets = cfg.feat_sets
        self.I = cfg.importance(self.cfg.input_size)

        feat_probs = []
        self.feat_distributions = []
        self.feat_set_offsets = [0]
        for i, size in enumerate(self.feat_sets):
            if i > 0:
                self.feat_set_offsets += [self.feat_set_offsets[-1] + self.feat_sets[i-1]]
            feat_probs.append([])
            for j in range(size):
                feat_probs[i].append(self.cfg.sparsity(j))
            feat_probs[i] = t.Tensor(feat_probs[i])
            feat_probs[i] /= t.sum(feat_probs[i]) #normalize
            self.feat_distributions.append(t.distributions.categorical.Categorical(feat_probs[i]))
        print('probabilities: {}'.format(feat_probs))
        

    def get_batch(self, batch_size: int = None,
                  identity=False,
                  ) -> Float[Tensor, "batch input_size"]:
        if identity:
            self.features = t.eye(self.cfg.input_size)
        else:
            if batch_size is None:
                batch_size = self.cfg.batch_size

            indices = [d.sample(sample_shape=(batch_size,1)) + o for d,o in zip(self.feat_distributions, self.feat_set_offsets)]
            self.features = t.zeros((batch_size, self.cfg.input_size))
            if self.cfg.correlate_set_magnitudes:
                randvals = t.rand(batch_size, 1)
            for idx in indices:
                if self.cfg.correlate_set_magnitudes:
                    self.features.scatter_(dim=1, index=idx, src=randvals)
                else:
                    self.features.scatter_(dim=1, index=idx, src=t.rand(batch_size, 1))
        return self.features.to(device)


class CorrelatedTMS(TMS):
    
    def __init__(self, cfg):
        super(CorrelatedTMS, self).__init__(cfg)
        if self.cfg.feat_sets is None:
            raise ValueError("Must set config feat_sets list for Composed Features Model")
        for i, s in enumerate(self.cfg.feat_sets):
            if s != self.cfg.feat_sets[0]:
                raise NotImplementedError("CorrelatedTMS only implemented for case where all feat_sets are the same length")
        self.feat_sets = cfg.feat_sets
        self.S = t.Tensor([cfg.sparsity(i) for i in range(len(self.feat_sets))])
        self.I = cfg.importance(self.cfg.input_size)


    def get_batch(self, batch_size: int = None,
                    identity = False,
                    ) -> Float[Tensor, "batch input_size"]:
        if identity:
            self.features = t.eye(self.cfg.input_size)
        else:
            if batch_size is None:
                batch_size = self.cfg.batch_size
            
            s = t.rand(batch_size, len(self.feat_sets))
            mask = t.repeat_interleave((s < self.S[None,:]).to(int), self.feat_sets[0], dim=-1).to(bool)
            self.features = t.rand(batch_size, self.cfg.input_size)
            self.features[mask] = 0 #make sparse
        return self.features.to(device)

class AntiCorrelatedTMS(CorrelatedTMS):

    def __init__(self, cfg):
        super(AntiCorrelatedTMS, self).__init__(cfg)
        if 2 != self.cfg.feat_sets[0]:
            raise NotImplementedError("AntiCorrelatedTMS only implemented for set length 2.")
    
    def get_batch(self, batch_size: int = None,
                    identity = False,
                    ) -> Float[Tensor, "batch input_size"]:
        self.features = super(AntiCorrelatedTMS, self).get_batch(batch_size, identity)
        if not identity:
            mask_int = t.stack([t.cat([t.randperm(self.feat_sets[0]) for _ in range(len(self.feat_sets))]) for _ in range(batch_size)]).to(device)
            self.features *= (mask_int == 0)
        return self.features.to(device)