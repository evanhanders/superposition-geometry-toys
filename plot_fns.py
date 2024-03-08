from dataclasses import dataclass
from IPython import display

import pathlib
import time
import h5py
import wandb

import torch as t
from torch import nn, Tensor
import torch.nn.functional as F
import torch.optim as optim
import einops
import numpy as np
from torch.distributions.categorical import Categorical

#Requires installation of https://github.com/krishnap25/geom_median
from geom_median.torch import compute_geometric_median

from typing import Optional, Callable, Union, List, Tuple, Dict
from jaxtyping import Float, Int, Bool

import matplotlib
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

@t.no_grad()
def w_cossim(W1, W2, sort=None):
    this_W1 = W1/W1.norm(dim=1, keepdim=True)
    this_W2 = W2/W2.norm(dim=1, keepdim=True)
    cossim = einops.einsum(this_W1, this_W2, "d_w1 d_hid, d_w2 d_hid -> d_w1 d_w2")
    if sort is None:
        argsort = t.argmax(cossim, dim=1)
    else:
        argsort = sort
    return cossim[:,argsort].detach().cpu(), argsort.detach().cpu()

@t.no_grad()
def w_enc(model, encoder, sort=None, instance=None):
    """
    Returns encoding-space vectors of a identitiy matrix of feature inputs.
        TODO: add monosemanticity_target (assumed = 1 right now).
    """
    #activations are ReLU((IW - b_dec)W_enc + b_enc)
    # = ReLU(W W_enc - b_dec W_enc + b_enc)
    term1 = einops.einsum(model.W, encoder.W_enc, 'd_in d_hid, n_inst d_hid d_sae -> n_inst d_in d_sae')
    if encoder.cfg.pre_encoder_bias:
        term2 = einops.einsum(encoder.b_dec, encoder.W_enc, 'n_inst d_hid, n_inst d_hid d_sae -> n_inst d_sae')[:,None,:]
    else:
        term2 = 0
    term3 = encoder.b_enc[:,None,:]
    actvecs = F.relu(term1 + term2 + term3)
    if instance is None:
        monosemanticity = encoder.measure_monosemanticity(model)
        instance = t.argmax(monosemanticity).item()
    if sort is None:
        # maybe try a weighted sum: maxs = top-k with k = 5 or something
        # take sum(maxs.values*max.indices)/sum(maxs.values) to get avg index.
        # sort by that weighted sum. Might give something semidiagonal?
        maxs = t.topk(actvecs[instance], dim=0, k=actvecs[instance].shape[0]//2)
        avg_index = t.sum(maxs.values*maxs.indices, dim=0)/t.sum(maxs.values, dim=0)
        argsort = t.argsort(avg_index)
        returnval = actvecs[instance,:, argsort]
    else:
        argsort = sort
        returnval = actvecs[instance,:, sort]
    return returnval.detach().cpu(), argsort.detach().cpu()

class TMSPlotter():
    def __init__(self, fig=None, axs = None):
        if fig is None:
            self.fig = plt.figure(figsize=(10,4))
        else:
            self.fig = fig
        self.axs = axs
        plt.ion() #interactive mode
        self.hdisplay = display.display("", display_id=True)
        self.im = None
        self.lines = None
        self.loss_lines = None
        self.initialized=False

        
    @t.no_grad()
    def update(self, model, autoencoder = None):
        if model.W.shape[1] == 2:
            #Initialize
            if not self.initialized:
                if self.axs is None:
                    ax1 = self.fig.add_subplot(1,2,1)
                    ax2 = self.fig.add_subplot(1,2,2)
                    self.axs = [ax1, ax2]
                elif len(self.axs) != 2:
                    raise ValueError("For 2 hidden dimensions, plotter must have 2 subplots.")
                
                self.axs[0].set_xlabel('hidden dim 1')
                self.axs[0].set_ylabel('hidden dim 2')
                self.axs[1].set_xlabel('iteration')
                if autoencoder is None:
                    self.axs[1].set_ylabel('loss')
                    self.axs[1].set_yscale('log')
                else:
                    self.axs[1].set_ylabel('monosemanticity')
            
                self.lines = []
                self.loss_lines = []
                feat_set_indx = 0
                if model.cfg.feat_sets is not None:
                    feat_set_steps = np.cumsum(model.cfg.feat_sets)
                    colors = ['g', 'b', 'orange', 'purple']
                else:
                    colors = ['k']
                for i in range(model.W.shape[0]):
                    if model.cfg.feat_sets is not None and feat_set_steps[feat_set_indx] == i: 
                        feat_set_indx += 1
                    xs = np.array((0, 1))
                    ys = np.array((0, 1))
                    self.lines.append(self.axs[0].plot(xs, ys, colors[feat_set_indx], '-')[0])
                if autoencoder is None:
                    self.loss_lines.append(self.axs[1].plot([], [], 'k-')[0])
                else:
                    for i in range(autoencoder.cfg.n_inst):
                         self.loss_lines.append(self.axs[1].plot([], [])[0])
                
                if autoencoder is not None:
                    for j in range(autoencoder.cfg.d_sae):
                        
                        sae_xs = np.array((0, autoencoder.W_dec[0,j,0].item()))
                        sae_ys = np.array((0, autoencoder.W_dec[0,j,1].item()))
                        self.lines.append(self.axs[0].plot(sae_xs, sae_ys, 'r-')[0])
                    #TODO: Track monosemanticity of each instance? Dead neuron number? Unclear.

                self.initialized = True

            #Update lines for plotting 
            for j in range(model.cfg.input_size):
                xs = np.array((0, model.W[j,0].item()))
                ys = np.array((0, model.W[j,1].item()))
                self.lines[j].set_xdata(xs)
                self.lines[j].set_ydata(ys)
                
            if autoencoder is not None:
                monosemanticity = autoencoder.measure_monosemanticity(model)
                instance = t.argmax(monosemanticity).item()
                for j in range(autoencoder.cfg.d_sae):
                    sae_xs = np.array((0, autoencoder.W_dec[instance,j,0].item()))
                    sae_ys = np.array((0, autoencoder.W_dec[instance,j,1].item()))

                    self.lines[model.cfg.input_size+j].set_xdata(sae_xs)
                    self.lines[model.cfg.input_size+j].set_ydata(sae_ys)
            
            if autoencoder is None and model.train_log is not None:
                self.loss_lines[0].set_xdata(model.train_log['steps'])
                self.loss_lines[0].set_ydata(model.train_log['losses'])
            elif autoencoder is not None and autoencoder.data_log is not None:
                for i in range(autoencoder.cfg.n_inst):
                    if len(autoencoder.data_log['step']) <= 1: break
                    self.loss_lines[i].set_xdata(autoencoder.data_log['step'])
                    self.loss_lines[i].set_ydata(np.array(autoencoder.data_log['monosemanticity'])[:,i])

            #Rescale and redraw
            self.axs[1].relim()
            self.axs[1].autoscale_view()

            this_max = 1.1*model.W.abs().max().item()
            self.axs[0].set_xlim(-this_max, this_max)
            self.axs[0].set_ylim(-this_max, this_max)

            self.fig.canvas.draw()
        else:
            #Get plot data
            if autoencoder is None:
                cossim, args = w_cossim(model.W, model.W)
            else:
                actvecs, argsacts = w_enc(model, autoencoder)
                monosemanticity = autoencoder.measure_monosemanticity(model)
                instance = t.argmax(monosemanticity).item()
                cossim, args = w_cossim(model.W, autoencoder.W_dec[instance], sort=argsacts)

            #Initialize
            if not self.initialized:
                if self.axs is None:
                    if autoencoder is None:
                        ax1 = self.fig.add_subplot(1,2,1)
                        ax2 = self.fig.add_subplot(1,2,2)
                        self.axs = [ax1, ax2]
                    else:
                        ax1 = self.fig.add_subplot(1,3,1)
                        ax2 = self.fig.add_subplot(1,3,2)
                        ax3 = self.fig.add_subplot(1,3,3)
                        self.axs = [ax1, ax2, ax3]
                elif (autoencoder is not None and len(self.axs) != 3) or (autoencoder is None and len(self.axs) != 2):
                    raise ValueError("Wrong number of subplot axes provided.")
                
                if autoencoder is not None:
                    self.axs[0].set_xlabel('SAE Features')
                    self.axs[0].set_ylabel('Data Features')
                    self.axs[1].set_xlabel('SAE Features')
                    self.axs[1].set_ylabel('Data Features')
                else:
                    self.axs[0].set_xlabel('Data Features')
                    self.axs[0].set_ylabel('Data Features')
                self.axs[-1].set_xlabel('iteration')
                if autoencoder is None:
                    self.axs[-1].set_ylabel('loss')
                    self.axs[-1].set_yscale('log')
                else:
                    self.axs[-1].set_ylabel('monosemanticity')
            
                self.loss_lines = []
                if autoencoder is None:
                    self.loss_lines.append(self.axs[-1].plot([], [], 'k-')[0])
                else:
                    for i in range(autoencoder.cfg.n_inst):
                         self.loss_lines.append(self.axs[-1].plot([], [])[0])
                
                if autoencoder is None:
                    self.im = self.axs[0].imshow(cossim.squeeze(), cmap='RdYlBu_r', vmin=-1, vmax=1)
                else:
                    self.im1 = self.axs[0].imshow(cossim.squeeze(), cmap='RdYlBu_r', vmin=-1, vmax=1)
                    self.im2 = self.axs[1].imshow(actvecs.squeeze(), cmap='viridis', vmin=0, vmax=1)
                    # self.axs[0].set_xlim(-0.5, min(cossim.shape)-0.5)
                    # self.axs[1].set_xlim(-0.5, min(actvecs.shape)-0.5)
                self.initialized = True
            else:
                #Update
                if autoencoder is None:
                    self.im.set_data(cossim.squeeze())
                    self.axs[0].set_xticks(range(len(args)), labels=[a.item() for a in args])
                else:
                    self.im1.set_data(cossim.squeeze())
                    self.axs[0].set_xticks(range(args.shape[0]), labels=[a.item() for a in args])
                    self.im2.set_data(actvecs.squeeze())
                    self.axs[1].set_xticks(range(argsacts.shape[0]), labels=[a.item() for a in argsacts])
                    # self.axs[0].set_xlim(-0.5, min(cossim.shape)-0.5)
                    # self.axs[1].set_xlim(-0.5, min(actvecs.shape)-0.5)
                
                if autoencoder is None and model.train_log is not None:
                    self.loss_lines[0].set_xdata(model.train_log['steps'])
                    self.loss_lines[0].set_ydata(model.train_log['losses'])
                elif autoencoder is not None and autoencoder.data_log is not None:
                    for i in range(autoencoder.cfg.n_inst):
                        if len(autoencoder.data_log['step']) <= 1: break
                        self.loss_lines[i].set_xdata(autoencoder.data_log['step'])
                        self.loss_lines[i].set_ydata(np.array(autoencoder.data_log['monosemanticity'])[:,i])

                #Rescale and redraw
                self.axs[-1].relim()
                self.axs[-1].autoscale_view()
        self.hdisplay.update(self.fig)

