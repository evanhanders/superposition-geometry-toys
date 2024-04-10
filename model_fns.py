from dataclasses import dataclass
import pathlib

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
    disallow_zeros: bool = False

    #composed features cfg options
    feat_sets: list = None
    active_features_per_draw: int = 1
    set_magnitude_correlation: float = 0
    correlated_feature_indices: list = None
    correlated_feature_boost: float = 0 #ranges from 0 to 1.
    onehot_probability: float = 0

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
        
        self.run_name = f"tms_nin{self.input_size}nhid{self.hidden_size}_numfeatsdraw{self.feat_set_size}_magnitude_corr{self.set_magnitude_correlation}_correlated_feature_boost{self.correlated_feature_boost}_onehot{self.onehot_probability:.2f}"


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
    
    def train(self, outdir: pathlib.Path = None):
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
        
        if outdir is not None:
            self.save(outdir, self.cfg.run_name)
        clear_output()

    def save(self, dir: pathlib.Path, run_name: str):
        """ Saves model weights, projection vector, and training dynamics """
        weights_fname = dir/f'weights_{run_name}.pt'
        t.save(self.state_dict(), weights_fname)

    def load(self, dir: pathlib.Path, run_name: str) -> dict:
        """ Loads model weights, projection vector, and training dynamics """
        weights_fname = dir/f'weights_{run_name}.pt'
        self.load_state_dict(t.load(weights_fname, map_location=device))

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
            sparse_mask = s < self.S[None,:]
            if self.cfg.disallow_zeros:
                zeros = sparse_mask.sum(dim=1) == 0
                max_indices = t.argmax(s, dim=1)
                sparse_mask[zeros][max_indices[zeros]] = True
            self.features = t.rand(batch_size, self.cfg.input_size)
            self.features[sparse_mask] = 0 #make sparse
        return self.features.to(device)


class ComposedFeatureTMS(TMS):

    def __init__(self, cfg):
        super(ComposedFeatureTMS, self).__init__(cfg)
        if self.cfg.feat_sets is None:
            raise ValueError("Must set config feat_sets list for Composed Features Model")
        self.feat_sets = cfg.feat_sets
        self.I = cfg.importance(self.cfg.input_size)

        #Uniform probability table.
        if self.cfg.onehot_probability > 0 and len(self.feat_sets) > 1:
            feat_probs = t.zeros([s + 1 for s in self.feat_sets])
        else:
            feat_probs = t.zeros(self.feat_sets)
        if len(self.feat_sets) == 1:
            feat_probs = t.Tensor([cfg.sparsity(i) for i in range(self.feat_sets[0])])
            self.feat_probs = feat_probs/t.sum(feat_probs)
            if self.cfg.correlated_feature_indices is not None:
                idx = self.cfg.correlated_feature_indices[0]
                prior_remaining_prob = (1-self.feat_probs[idx])
                self.feat_probs[idx] += self.cfg.correlated_feature_boost*(prior_remaining_prob)
                remaining_prob = (1-self.feat_probs[idx])
                for i in range(self.feat_sets[0]):
                    if i != idx:
                        self.feat_probs[i] *= remaining_prob/prior_remaining_prob
            self.feat_indices = t.arange(self.feat_sets[0])
        elif len(self.feat_sets) == 2:
            baseline_0 = t.Tensor([cfg.sparsity(i) for i in range(self.feat_sets[0])])
            baseline_1 = t.Tensor([cfg.sparsity(i) for i in range(self.feat_sets[1])])
            feat_probs[:self.feat_sets[0],0] = baseline_0
            for i in range(self.feat_sets[0]):
                feat_probs[i,:self.feat_sets[1]] = baseline_1 * (baseline_0[i] / baseline_1[0])
            self.feat_probs = feat_probs/t.sum(feat_probs) #normalizeself.feat_probs = feat_probs.ravel()
            if self.cfg.correlated_feature_indices is not None:
                original_feat_probs = self.feat_probs.clone()
                (idx0, idx1) = self.cfg.correlated_feature_indices
                rowprob = self.feat_probs[idx0].sum()
                colprob = self.feat_probs[:,idx1].sum()
                prior_remaining_rowprob = rowprob-self.feat_probs[idx0,idx1]
                prior_remaining_colprob = colprob-self.feat_probs[idx0,idx1]
                prior_remaining_prob = min(prior_remaining_rowprob, prior_remaining_colprob)
                self.feat_probs[idx0,idx1] += self.cfg.correlated_feature_boost*(prior_remaining_prob)
                remaining_rowprob = (rowprob-self.feat_probs[idx0,idx1]).sum()
                remaining_colprob = (colprob-self.feat_probs[idx0,idx1]).sum()
                
                #update messed-with feature's row and column probabilities
                for i in range(self.feat_sets[1]):
                    if i != idx1:
                        self.feat_probs[idx0,i] *= remaining_colprob/prior_remaining_colprob
                for j in range(self.feat_sets[0]):
                    if j != idx0:
                        self.feat_probs[j,idx1] *= remaining_rowprob/prior_remaining_rowprob

                #update off-row probabilities
                for i in range(self.feat_sets[0]):
                    if i == idx0: continue
                    new_rowprob = self.feat_probs[i].sum()
                    adjustable_rowprob = new_rowprob - self.feat_probs[i,idx1]
                    unaccounted_prob = (original_feat_probs[i].sum()-new_rowprob)

                    factor = 1 + unaccounted_prob/adjustable_rowprob
                    for j in range(self.feat_sets[1]):
                        if j == idx1: continue
                        self.feat_probs[i,j] *= factor

                # Adjust for one-hot-ness
                if self.cfg.onehot_probability > 0:
                    self.feat_probs *= (1-self.cfg.onehot_probability)
                    norm_baseline_0 = baseline_0/t.sum(baseline_0)
                    norm_baseline_1 = baseline_1/t.sum(baseline_1)
                    self.feat_probs[:self.feat_sets[0],-1] = self.cfg.onehot_probability*norm_baseline_0*self.feat_sets[0]/np.sum(self.feat_sets)
                    self.feat_probs[-1,:self.feat_sets[1]] = self.cfg.onehot_probability*norm_baseline_1*self.feat_sets[1]/np.sum(self.feat_sets)
            
            if self.cfg.onehot_probability > 0 and len(self.feat_sets) > 1:
                if len(self.feat_sets) > 2:
                    raise NotImplementedError("One-hot probability only implemented for 2 feature sets")
                flat_feat_indices = t.Tensor([(i // self.feat_sets[0], self.feat_sets[0] + i % self.feat_sets[0]) for i in range(np.prod(self.feat_sets))]).long()

                self.feat_indices = t.zeros((self.feat_probs.shape[0], self.feat_probs.shape[1], 2))
                self.feat_indices[:self.feat_sets[0], :self.feat_sets[1],:] = flat_feat_indices.reshape(self.feat_sets[0], self.feat_sets[1], 2)
                self.feat_indices[:,-1,0] = self.feat_indices[:,0, 0]
                self.feat_indices[-1,:,1] = self.feat_indices[0,:,1]
                self.feat_indices[:,-1,1] = self.feat_indices[:,-1,0]
                self.feat_indices[-1,:,0] = self.feat_indices[-1,:,1]
                
                self.feat_probs = self.feat_probs.ravel()
                self.feat_indices = self.feat_indices.reshape(-1,2)
                # print(self.feat_indices[:,:,0])
                # print(self.feat_indices[:,:,1])
            else:
                self.feat_probs = self.feat_probs.ravel()
                self.feat_indices = t.Tensor([(i // self.feat_sets[0], self.feat_sets[0] + i % self.feat_sets[0]) for i in range(self.feat_probs.shape[0])]).long()
            # print(self.feat_probs, self.feat_indices)
            
        else:
            raise NotImplementedError("ComposedFeatureTMS only implemented for case with <= 2 feature sets")
        self.print_prob_table()
        self.feat_sampler = t.distributions.categorical.Categorical(self.feat_probs)


    def get_prob_table(self):
        if self.cfg.onehot_probability > 0:
            return self.feat_probs.reshape([i+1 for i in self.cfg.feat_sets])
        else:
            return self.feat_probs.reshape(self.cfg.feat_sets)

    def print_prob_table(self):
        prob_table = self.get_prob_table()
        if len(self.cfg.feat_sets) > 2 and len(self.cfg.fet_sets) > 0:
            raise NotImplementedError("Can only output 1- or 2D probability table")

        xs = [f'y{i+1}' for i in range(self.cfg.feat_sets[0])]
        header = f'{"":<10s}' + (('{:<10s}')*len(xs)).format(*tuple(xs))
        if self.cfg.onehot_probability > 0:
            header += f'{"No yi":<10s}'
        print(header)
        if len(self.cfg.feat_sets) == 1:
            string = f'{"probs:":<10s}' + (('{:<10.4f}')*prob_table.shape[1]).format(*tuple(prob_table))
            print(string)
        else:
            ys = [f'x{i+1}' for i in range(self.cfg.feat_sets[1])]
            if self.cfg.onehot_probability > 0:
                ys += [f'{"No xi":<10s}']
            for y, row in zip(ys, prob_table):
                string = f'{y:<10s}' + (('{:<10.4f}')*prob_table.shape[1]).format(*tuple(row))
                print(string)

    def set_prob_table(self, prob_table):
        #check if it's a valid table
        if not np.isclose(prob_table.sum().item() - 1, 0, atol=1e-6):
            raise ValueError("A probability table must sum to 1.")

        if prob_table.numel() != self.feat_probs.numel():
            raise ValueError(f"Mismatch between probability table and expected size: prob table has {prob_table.size} elements but needs {self.feat_probs.size}")

        self.feat_probs = prob_table.ravel()
        self.feat_sampler = t.distributions.categorical.Categorical(self.feat_probs)
        print('setting new probability table:')
        self.print_prob_table()
        
    def get_batch(self, batch_size: int = None,
                  identity=False,
                  ) -> Float[Tensor, "batch input_size"]:
        if identity:
            self.features = t.eye(self.cfg.input_size)
        else:
            if batch_size is None:
                batch_size = self.cfg.batch_size

            self.features = t.zeros((batch_size, self.cfg.input_size))
            randvals = t.rand(batch_size, len(self.feat_sets), self.cfg.active_features_per_draw)

            f = self.cfg.set_magnitude_correlation
            #if f = 0, then keep current value; if f = 1, then use set 0 value.
            for i in range(len(self.feat_sets) - 1):
                randvals[:,i+1] = (1-f)*randvals[:,i+1] + f*randvals[:,0]


            for draw in range(self.cfg.active_features_per_draw):
                idx = self.feat_sampler.sample(sample_shape=(batch_size,))
                indices = self.feat_indices[idx][...,None]

                feats = t.zeros((batch_size, self.cfg.input_size))

                if len(self.feat_sets) == 1:
                    feats.scatter_(dim=1, index=indices, src=randvals[...,draw])
                else:
                    for i in range(indices.shape[1]):
                        idx = indices[:,i]
                        feats.scatter_(dim=1, index=idx, src=randvals[:,i,draw][:,None])
                self.features += feats
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