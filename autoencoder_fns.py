"""
Code for the AutoEncoder class, which is a sparse autoencoder with L1 regularization for
dictionary learning of toy model neurons.

This code is largely based on Joseph Bloom's implementation, which can be found here: https://github.com/jbloomAus/mats_sae_training/tree/main
"""

from IPython.display import display, clear_output
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

from plot_fns import TMSPlotter
from tqdm.notebook import tqdm

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

DTYPES = {"fp32": t.float32, "fp16": t.float16, "bf16": t.bfloat16}

@dataclass
class AutoEncoderConfig:
    #SAE Parameters
    b_dec_init_method = "geometric_median"
    pre_encoder_bias: bool = False
    normalize_in_l2: bool = True
    normalize_to_lesseq_one: bool = False
    n_inst: int = 1
    dict_mult: int = 8
    d_in: int = 1024
    enc_dtype: str = "fp32"

    #Training Parameters
    seed: int = None
    training_samples: int = 128_000_000
    batch_size: int = 1024
    lr: float = 3e-4
    l1_coeff: float = 1e-1
    lr_scheduler_name: str = "constant_with_warmup_and_cooldown" # "constant" "constant_with_warmup" "constant_with_warmup_and_cooldown"
    lr_warmup_frac: int = 0.1
    lr_cooldown_frac: float = 0.2
    adam_beta1: float = 0
    adam_beta2: float = 0.999

    #Janky training stuff
    dropout_prob: int = 0
    do_random_penalty: bool = False 
    penalization_weights: list = None #TODO: define with default_factory
    noise_scale: float = 0

    #Resampling protocol
    use_resampling: bool = False
    neuron_resample_window: int = 100
    neuron_resample_scale: float = 0.5
    use_ghost_grads: bool = True
    ghost_grads_cooldown: int = 1000
    dead_feature_window: int = 400  # unless this window is larger feature sampling,
    dead_feature_threshold: float = 1e-6
    ghost_revivals: bool = False
    ghost_revival_count: int = 5

    # WANDB
    log_to_wandb: bool = False
    wandb_project: str = "toy-models-superposition"

    def __post_init__(self):
        if self.seed is None:
            self.seed = int(np.random.rand() * (2**32 - 1))
        self.dtype = DTYPES[self.enc_dtype]
        self.d_sae = int(np.round(self.d_in * self.dict_mult))
        self.batches = int(np.ceil(self.training_samples / self.batch_size))

        self.lr_warmup_steps = int(self.lr_warmup_frac*self.batches)
        self.lr_cooldown_steps = int(self.lr_cooldown_frac*self.batches)
        self.lr_steps_before_cooldown = self.batches - self.lr_cooldown_steps

        self.run_name = f"ninst{self.n_inst}_din{self.d_in}_dsae{self.d_sae}-L1-{self.l1_coeff}-LR-{self.lr}-Seed-{self.seed}"

        if self.b_dec_init_method not in ["geometric_median", "zeros"]:
            raise ValueError(
                f"b_dec_init_method must be geometric_median or zeros. Got {self.b_dec_init_method}"
            )
        elif self.b_dec_init_method == "zeros":
            print(
                "Warning: We are initializing b_dec to zeros. This is probably not what you want."
            )

        if self.use_resampling and self.use_ghost_grads:
            raise ValueError("Cannot use both resampling and ghost grads")
    
        if self.do_random_penalty and self.penalization_weights is None:
            self.penalization_weights = [0.5, 1]

class AutoEncoder(nn.Module):
    """
    Sparse Autoencoder with L1 regularization and ghost gradient implementation.
    """
    def __init__(self, cfg: AutoEncoderConfig):
        super().__init__()

        self.cfg = cfg
        self.training = False
        t.manual_seed(cfg.seed)
        self.l1_coeff = self.cfg.l1_coeff
        self.data_log = None

        self.W_enc = nn.Parameter(t.nn.init.kaiming_uniform_(t.empty(cfg.n_inst, cfg.d_in, cfg.d_sae, dtype=cfg.dtype)))
        self.b_enc = nn.Parameter(t.zeros((cfg.n_inst, cfg.d_sae), dtype=cfg.dtype))
        self.W_dec = nn.Parameter(t.nn.init.kaiming_uniform_(t.empty(cfg.n_inst, cfg.d_sae, cfg.d_in, dtype=cfg.dtype)))
        self.b_dec = nn.Parameter(t.zeros((cfg.n_inst, cfg.d_in), dtype=cfg.dtype))
        self.normalize_decoder()
        

    @t.no_grad()
    def normalize_decoder(self) -> None:
        """
        Normalizes the decoder weights to have unit norm.
        """
        #old:
        # self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True)

        #new:
        norm = self.W_dec.data.norm(dim=2, keepdim=True)
        if self.cfg.normalize_to_lesseq_one:
            norm[norm < 1] = 1
        self.W_dec.data /= norm

    def forward(self, x: Float[Tensor, 'batch d_in'],
                dead_neuron_mask: Bool[Tensor, "n_inst d_sae"] = None,
                ghost_grad_cooldown: Float[Tensor, "n_inst d_sae"] = None
    )-> Tuple[Float[Tensor, '1'], Float[Tensor, 'n_inst batch d_in'], Float[Tensor, 'n_inst batch d_sae'], Float[Tensor, '1'], Float[Tensor, '1'], Float[Tensor, '1']]:
        """
        Forward autoencoder pass. The autoencoder tries to reconstruct the input x.
        During training, the autoencoder also tries to minimize the L1 loss.
        Dead neurons are resampled using the Ghost protocol described in Anthropic's january update:
          https://transformer-circuits.pub/2024/jan-update/index.html#dict-learning-resampling
          -> Comments are copied directly from there.

        Returns a tuple of:
        - loss: The total loss
        - x_reconstruct: The reconstructed input
        - acts: The activations of the SAE hidden layer
        - mse_loss: The mean squared error loss
        - l1_loss: The L1 loss
        - mse_loss_ghost_resid: The mean squared error loss of the ghost residuals
        """
        x = x.to(self.cfg.dtype)
        noise = t.normal(0, x.std()*self.cfg.noise_scale, size=x.shape, device=device, dtype=self.cfg.dtype)
        # x = x + noise
        if self.cfg.pre_encoder_bias:
            x_cent = x[None,:] - self.b_dec[:,None,:]
            pre = einops.einsum(x_cent + noise, self.W_enc, "n_inst batch d_in, n_inst d_in d_sae -> n_inst batch d_sae") + self.b_enc[:,None,:]
        else:
            pre = einops.einsum(x + noise, self.W_enc, "batch d_in, n_inst d_in d_sae -> n_inst batch d_sae") + self.b_enc[:,None,:]
        acts = F.relu(pre)
        if self.cfg.dropout_prob > 0:
            acts = F.dropout(acts, p=self.cfg.dropout_prob, training=self.training)
        x_reconstruct = einops.einsum(acts, self.W_dec, "n_inst batch d_sae, n_inst d_sae d_in -> n_inst batch d_in") + self.b_dec[:,None,:]

        if self.cfg.normalize_in_l2:
            x_norm = t.norm(x, dim=-1, keepdim=True)
            # print(x_norm)
            x_norm.data[x_norm == 0] = 1
            x_norm = x_norm[None,:]
            # x_norm *= np.sqrt(self.cfg.d_in)
        else:
            x_norm = 1

        mse_loss = ((x_reconstruct - x)/x_norm).pow(2).sum(dim=-1, keepdim=True)#.sqrt() #l2 norm of each sample w/ sqrt.
        mse_loss_ghost_resid = t.zeros_like(mse_loss)#t.tensor(0, dtype=self.cfg.dtype, device=device)

        if self.cfg.use_ghost_grads and self.training and (dead_neuron_mask is not None) and dead_neuron_mask.any():

            for i in range(self.cfg.n_inst):
                if not dead_neuron_mask[i].any():
                    continue
                
                # print(mse_loss)

                #Ghost protocol
                # 1. Compute the reconstruction residuals and the MSE loss as normal.
                residual = (x - x_reconstruct[i])/x_norm
                # residual = (x_reconstruct - x)/x_norm
                l2_norm_residual = t.norm(residual, dim=-1)

                # 2. Compute a second forward pass of the autoencoder using just the dead neurons.
                # In this forward pass, we replace the ReLU activation function on the dead neurons with an exponential activation function.
                # Joseph note: We don't need b_dec here since we're not reconstructing the input, just the residual.
                acts_dead_neurons = t.exp(pre[i,:,dead_neuron_mask[i]])
                ghost_out = acts_dead_neurons @ self.W_dec[i,dead_neuron_mask[i]]

                # 3. Scale the output of the dead neurons so that the L2 norm is ½ the L2 norm of the autoencoder residual from (1).
                # Note that the scale factor is treated as a constant for gradient propagation purposes.
                l2_norm_ghost_out = t.norm(ghost_out, dim=-1)
                norm_scaling_factor = l2_norm_residual / (1e-6 + 2* l2_norm_ghost_out)
                ghost_out = ghost_out * norm_scaling_factor.unsqueeze(-1).detach()

                # 4. Compute the MSE between the autoencoder residual and the output from the dead neurons.
                mse_loss_ghost_resid_tmp = (residual - ghost_out).pow(2).sum(dim=-1, keepdim=True)#.sqrt() #l2 norm w/ sqrt

                # 5. Rescale that MSE to be equal in magnitude to the normal reconstruction loss from step 1.
                # The normal reconstruction loss is treated as a constant in this step for gradient propagation purposes.
                mse_rescaling_factor = (mse_loss[i] / (1e-6 + mse_loss_ghost_resid_tmp)).detach()
                mse_loss_ghost_resid[i] = mse_loss_ghost_resid_tmp * mse_rescaling_factor

        # 6. Add ghost result to the total loss.
        mse_loss_ghost_resid = mse_loss_ghost_resid#.mean() #mean over samples
        mse_loss = mse_loss#.mean() #mean over samples
        if ghost_grad_cooldown is not None:
            sparsity = t.abs(ghost_grad_cooldown[:,None,:]*acts).sum(dim=-1, keepdim=True)#.mean(dim=(0,))
        else:
            sparsity = t.abs(acts).sum(dim=-1, keepdim=True)

        if self.training and self.cfg.do_random_penalty:
            penalty_weights = self.cfg.penalization_weights[0] + self.cfg.penalization_weights[1]*t.rand_like(sparsity)
            sparsity = sparsity * penalty_weights       

        l1_loss = self.l1_coeff * sparsity
        loss = (mse_loss + l1_loss + mse_loss_ghost_resid).mean()
        return loss, x_reconstruct, acts, mse_loss.mean(), l1_loss.mean(), mse_loss_ghost_resid.mean()

    @t.no_grad()
    def initialize_b_dec(self, *args):
        """ Initialization method for b_dec. Used in train(). """
        if self.cfg.b_dec_init_method == "geometric_median":
            self.initialize_b_dec_with_geometric_median(*args)
        elif self.cfg.b_dec_init_method == "zeros":
            pass
        else:
            raise ValueError(f"Unexpected b_dec_init_method: {self.cfg.b_dec_init_method}")

    @t.no_grad()
    def initialize_b_dec_with_geometric_median(self, activations: Float[Tensor, "batch d_in"]):
        """
        Reinitializes b_dec with the geometric median of the activations.
        Relies on the geom_median package:
            https://github.com/krishnap25/geom_median
        """

        previous_b_dec = self.b_dec.clone().cpu()
        all_activations = activations.detach().cpu() #account for instances.
        out = compute_geometric_median(
                all_activations,
                skip_typechecks=True,
                maxiter=100, per_component=False).median


        out = t.tensor(out, dtype=self.cfg.dtype)

        for i in range(self.cfg.n_inst):

            previous_distances = t.norm(all_activations - previous_b_dec[i], dim=-1)
            distances = t.norm(all_activations - out, dim=-1)

            print("Reinitializing b_dec with geometric median of activations")
            print(f"Previous distances: {previous_distances.median(0).values.mean().item()}")
            print(f"New distances: {distances.median(0).values.mean().item()}")
            self.b_dec.data[i,:] = out.to(device)

    @t.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Update grads to remove the parallel component of the gradient with respect to the decoder weights.
        TODO: Should I make sure I'm dotting onto unit vectors?
        """
        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "n_inst d_sae d_in, n_inst d_sae d_in -> n_inst d_sae",
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "n_inst d_sae, n_inst d_sae d_in -> n_inst d_sae d_in",
        )

    def measure_monosemanticity(self, model, delta=1e-10, threshold=0.05):
        """
        Measures monosemanticity of neurons using two metrics:
            -monosemanticity1 is the metric described in eqn 7 of https://arxiv.org/abs/2211.09169
            -monosemanticity2 uses a cutoff threshold to see how many features each neuron activates on.
        """
        batch = model.get_batch(identity=True)
        output = model.forward(batch)
        loss, x_reconstruct, acts, mse_loss, l1_loss, mse_loss_ghost_resid = self.forward(model.h)

        maxfeat = t.max(acts, dim=1).values #max over feature dim
        sumfeat = delta + t.sum(acts, dim=1) #sum of activations over all featuers
        monosemanticity = t.sum(maxfeat/sumfeat, dim=-1) / self.cfg.d_sae #t.sum(sumfeat > 10*delta)
        return monosemanticity

    @t.no_grad()
    def resample_neurons(
        self,
        h: Float[Tensor, "batch_size d_in"],
        dead_neurons_mask: Float[Tensor, "n_inst d_sae"],
        neuron_resample_scale: float,
    ) -> None:
        '''
        Resamples neurons that have been dead for `dead_neuron_window` steps, according to `frac_active`.
        '''
        loss, h_reconstruct, acts, mse_loss, l1_loss, mse_loss_ghost_resid = self.forward(h)

        # Create an object to store the dead neurons (this will be useful for plotting)
        dead_neurons_mask = t.empty((self.cfg.d_sae,), dtype=t.bool, device=self.W_enc.device)


        n_dead = dead_neurons_mask.sum(dim=-1)
        for instance in range(self.cfg.n_inst):
            dead_neurons = t.where(dead_neurons_mask[instance])[0]
            # print('resampling {}'.format(dead_neurons))
            if n_dead == 0:
                return
            elif n_dead == self.cfg.d_sae:
                nn.init.kaiming_uniform_(self.W_enc)
                self.W_enc[:] *= neuron_resample_scale
                #TODO: set according to initialization policy
                self.b_enc[:] = 0
                return
            this_l2_loss = (h_reconstruct[instance] - h).pow(2).sum(-1)
            probs = this_l2_loss / this_l2_loss.sum()
            distribution = Categorical(probs=probs)
            dead_indices = distribution.sample((n_dead,))
            weight_scale = self.W_enc[instance,:,t.where(t.logical_not(dead_neurons_mask[instance]))[0]].norm(dim=-1).mean()
            for samp_ind, neur_ind in zip(dead_indices, dead_neurons):
                centered = h[samp_ind] - self.b_dec
                centered /= centered.norm()
                centered *= weight_scale
                self.W_enc[instance,:,neur_ind] = neuron_resample_scale*centered
            self.b_enc[instance,dead_neurons] = 1


    def train(
        self,
        model,
        log_bump: float = 1.02,
        plot=True,
        plot_cadence=100,
        rootdir: pathlib.Path = pathlib.Path('./')
        ):
        """
        Optimizes the autoencoder using the given hyperparameters.
        Saves autoencoder using the save() function at the end.

        This function should take a trained toy model as input.
        """
        #Activate training for ghost grads
        self.training = True

        # track active features
        act_freq_scores = t.zeros((self.cfg.n_inst, self.cfg.d_sae), device=device)
        n_forward_passes_since_fired = t.zeros((self.cfg.n_inst, self.cfg.d_sae), device=device)
        ghost_grads_cooldown = t.zeros((self.cfg.n_inst, self.cfg.d_sae), dtype=t.long).to(device)
        ghost_revival_counter = t.zeros((self.cfg.n_inst, self.cfg.d_sae), dtype=t.long).to(device)
        ghost_revival_toggle = t.zeros((self.cfg.n_inst, self.cfg.d_sae), dtype=t.bool).to(device)
        cooldown = t.ones_like(ghost_grads_cooldown)
        n_frac_active_tokens = 0

        optimizer = t.optim.Adam(list(self.parameters()), lr=self.cfg.lr, betas=(self.cfg.adam_beta1, self.cfg.adam_beta2))

        if self.cfg.lr_scheduler_name == "constant_with_warmup_and_cooldown":
            scheduler1 = t.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: min(1.0, (step + 1) / self.cfg.lr_warmup_steps),
            )
            scheduler2 = t.optim.lr_scheduler.LinearLR(optimizer,
                    start_factor=1.0,
                    end_factor=0.0,
                    total_iters=self.cfg.lr_cooldown_steps
            )
            scheduler = t.optim.lr_scheduler.SequentialLR(optimizer,
                    schedulers=[scheduler1, scheduler2],
                    milestones=[self.cfg.lr_steps_before_cooldown]
            )
        elif self.cfg.lr_scheduler_name == "constant_with_warmup":
            scheduler = t.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: min(1.0, (step + 1) / self.cfg.lr_warmup_steps),
            )
        else:
            scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

        if self.cfg.log_to_wandb:
            wandb.init(project=self.cfg.wandb_project, name=self.cfg.run_name)
            wandb.watch(model, log='all', log_freq=1000)

        #All models have feature sparsity >= 1/512, so 5000 samples is a robust batch
        large_batch_out = model(model.get_batch(5000)) #TODO implement get_batch.
        self.initialize_b_dec(model.h)
        del large_batch_out

        if plot:
            plotter = TMSPlotter()

        # Create lists to store data we'll eventually be plotting
        self.data_log = {
            "step": [],
            "mse_loss": [],
            "l1_loss": [],
            "ghost_grad_loss": [],
            "overall_loss": [],
            "explained_neuron_variance": [],
            "explained_neuron_variance_std": [],
            "explained_output_variance": [],
            "explained_output_variance_std": [],
            "l0": [],
            "strong_l0": [],
            "monosemanticity": [],
            "mean_passes_since_fired": [],
            "n_passes_since_fired_over_threshold": [],
            "below_1e-5": [],
            "below_1e-6": [],
            "dead_features": [],
            "ghost_grad_neurons": [],
            "n_training_tokens": [],
            "current_learning_rate": [],
            "log10_l1_coeff": [],
            }

        progress_bar = tqdm(total = self.cfg.batches)
        output_batch = 1
        for step in range(self.cfg.batches):
            # Get a batch of hidden activations from the model
            # with t.no_grad():
            with t.no_grad():
                batch = model.get_batch(self.cfg.batch_size)
                output = model(batch)
                h = model.h

            # Normalize the decoder weights before each optimization step
            self.normalize_decoder()

            #Get dead neuron mask
            dead_neuron_mask = (n_forward_passes_since_fired > self.cfg.dead_feature_window).bool()
            ghost_grads_cooldown[dead_neuron_mask] = self.cfg.ghost_grads_cooldown
            ghost_grads_cooldown[t.logical_not(dead_neuron_mask)] -= 1
            ghost_grads_cooldown[ghost_grads_cooldown < 0] = 0


            # Revive ghost neurons
            revival_mask = (ghost_revival_counter == self.cfg.ghost_revival_count)
            if self.cfg.ghost_revivals and revival_mask.any():
                # Resample
                self.resample_neurons(h, revival_mask, self.cfg.neuron_resample_scale)
                ghost_revival_counter[revival_mask] = 0

            # Resample dead neurons
            if self.cfg.use_resampling and ((step + 1) % self.cfg.neuron_resample_window == 0):
                # Resample
                self.resample_neurons(h, dead_neuron_mask, self.cfg.neuron_resample_scale)

            # Optimize
            optimizer.zero_grad()
            if self.cfg.ghost_grads_cooldown > 0:
                cooldown[:] = 1 - ghost_grads_cooldown/self.cfg.ghost_grads_cooldown

            #If a feature is toggled off (it was alive on previous iteration) but it's now dead (ghost_mask=True), add 1 to ghost revival counter
            ghost_revival_counter[t.logical_and(t.logical_not(ghost_revival_toggle), dead_neuron_mask)] += 1
            #note that this feature is turned on for the future.
            ghost_revival_toggle[dead_neuron_mask] = True
            loss, h_reconstruct, acts, mse_loss, l1_loss, mse_loss_ghost_resid = self.forward(h, dead_neuron_mask, cooldown)


            #
            did_fire = ((acts > self.cfg.dead_feature_threshold).float().sum(1) > 0)
            n_forward_passes_since_fired += 1
            n_forward_passes_since_fired[did_fire] = 0

            loss.backward()
            self.remove_gradient_parallel_to_decoder_directions()
            optimizer.step()
            scheduler.step()

            progress_bar.update()

            with t.no_grad():
                if plot and (step+1) % plot_cadence == True:
                    plotter.update(model, autoencoder=self)
                # Calculate the sparsities, and add it to a list, calculate sparsity metrics
                act_freq_scores += (acts.abs() > 0).float().sum(1)
                n_frac_active_tokens += self.cfg.batch_size
                feature_sparsity = act_freq_scores / n_frac_active_tokens
                if (step+1) == output_batch or (step + 1) == self.cfg.batches:


                    #Get true measure of dead features.
                    term1 = einops.einsum(model.W, self.W_enc, 'd_in d_hid, n_inst d_hid d_sae -> n_inst d_in d_sae')
                    if self.cfg.pre_encoder_bias:
                        term2 = einops.einsum(self.b_dec, self.W_enc, 'n_inst d_hid, n_inst d_hid d_sae -> n_inst d_sae')[:,None,:]
                    else:
                        term2 = 0
                    term3 = self.b_enc[:,None,:]
                    actvecs = F.relu(term1 + term2 + term3)
                    dead_features = list(t.sum(actvecs.sum(dim=1) == 0, dim=-1).detach().cpu().numpy())

                    monosemanticity = self.measure_monosemanticity(model)
                    output_batch = int(np.ceil(log_bump*output_batch))
                    l0 = (acts > 0).float().sum(-1).mean(-1)
                    strong_l0 = (acts > acts.max()*1e-2).float().sum(-1).mean(-1)
                    current_learning_rate = optimizer.param_groups[0]["lr"]

                    #finish forward pass from activation
                    reconst_z = t.einsum('ih,nbh->nbi', model.W, h_reconstruct) + model.b
                    reconst_output = F.relu(reconst_z)
                    reconst_output_variance = (reconst_output - output).pow(2).sum(-1)
                    total_output_variance = (output+1e-8).pow(2).sum(-1)
                    explained_output_variance = 1 - reconst_output_variance/total_output_variance

                    reconst_neuron_variance = (h_reconstruct - h).pow(2).sum(dim=-1)
                    total_neuron_variance = (h+1e-8).pow(2).sum(dim=-1)
                    explained_neuron_variance = 1 - reconst_neuron_variance/total_neuron_variance


                    self.data_log["step"].append(step+1)
                    self.data_log["mse_loss"].append(mse_loss.item())
                    self.data_log["l1_loss"].append(l1_loss.item() / self.l1_coeff) # normalize by l1 coefficient
                    self.data_log["ghost_grad_loss"].append(mse_loss_ghost_resid.item())
                    self.data_log["overall_loss"].append(loss.item())
                    self.data_log["explained_neuron_variance"].append(explained_neuron_variance.mean(dim=-1).detach().cpu().numpy())
                    self.data_log["explained_neuron_variance_std"].append(explained_neuron_variance.std(dim=-1).detach().cpu().numpy())
                    self.data_log["explained_output_variance"].append(explained_output_variance.mean(dim=-1).detach().cpu().numpy())
                    self.data_log["explained_output_variance_std"].append(explained_output_variance.std(dim=-1).detach().cpu().numpy())
                    self.data_log["l0"].append(l0.detach().cpu().numpy())
                    self.data_log["strong_l0"].append(strong_l0.detach().cpu().numpy())
                    self.data_log["monosemanticity"].append(monosemanticity.detach().cpu().numpy())
                    self.data_log["mean_passes_since_fired"].append(n_forward_passes_since_fired.mean(dim=-1).detach().cpu().numpy())
                    self.data_log["n_passes_since_fired_over_threshold"].append(dead_neuron_mask.sum(dim=-1).detach().cpu().numpy())
                    self.data_log["below_1e-5"].append((feature_sparsity < 1e-5).float().mean(dim=-1).detach().cpu().numpy())
                    self.data_log["below_1e-6"].append((feature_sparsity < 1e-6).float().mean(dim=-1).detach().cpu().numpy())
                    self.data_log["dead_features"].append((feature_sparsity < self.cfg.dead_feature_threshold).float().mean(dim=-1).detach().cpu().numpy())
                    self.data_log['ghost_grad_neurons'].append((dead_neuron_mask).sum(dim=-1).detach().cpu().numpy())
                    self.data_log["n_training_tokens"].append(n_frac_active_tokens)
                    self.data_log["current_learning_rate"].append(current_learning_rate)
                    self.data_log["log10_l1_coeff"].append(np.log10(self.l1_coeff))

                    progress_bar.set_description(f"Batch {step+1}; lr {self.data_log['current_learning_rate'][-1]:.1e} Loss: {(loss.item()):.2e} (L1: {l1_loss.item():.2e}, L2: {mse_loss.item():.2e}, G: {mse_loss_ghost_resid.item():.2e}), mono: {self.data_log['monosemanticity'][-1].max().item():.4f}, dead: {self.data_log['ghost_grad_neurons'][-1]} | {dead_features}")


                    if self.cfg.log_to_wandb:
                        # metrics for currents acts
                        wandb.log(
                            {
                                #losses
                                "losses/mse_loss": self.data_log["mse_loss"][-1],
                                "losses/l1_loss": self.data_log["l1_loss"][-1],
                                "losses/ghost_grad_loss": self.data_log["ghost_grad_loss"][-1],
                                "losses/overall_loss": self.data_log["overall_loss"][-1],
                                # variance explained
                                "metrics/explained_neuron_variance": self.data_log["explained_neuron_variance"][-1],
                                "metrics/explained_neuron_variance_std": self.data_log["explained_neuron_variance_std"][-1],
                                "metrics/explained_output_variance": self.data_log["explained_output_variance"][-1],
                                "metrics/explained_output_variance_std": self.data_log["explained_output_variance_std"][-1],
                                "metrics/l0": self.data_log["l0"][-1],
                                "metrics/strong_l0": self.data_log["strong_l0"][-1],
                                # sparsity
                                "sparsity/mean_passes_since_fired": self.data_log["mean_passes_since_fired"][-1],
                                "sparsity/n_passes_since_fired_over_threshold": self.data_log["n_passes_since_fired_over_threshold"][-1],
                                "sparsity/below_1e-5": self.data_log["below_1e-5"][-1],
                                "sparsity/below_1e-6": self.data_log["below_1e-6"][-1],
                                "sparsity/dead_features": self.data_log["dead_features"][-1],
                                "sparsity/ghost_grad_neurons": self.data_log['ghost_grad_neurons'][-1],
                                "sparsity/mono_semanticity1": self.data_log["monosemanticity"][-1],
                                # details
                                "details/n_training_tokens": self.data_log["n_training_tokens"][-1],
                                "details/current_learning_rate": self.data_log["current_learning_rate"][-1],
                                "details/log10_l1_coeff": self.data_log["log10_l1_coeff"][-1],
                            },
                            step=step+1,
                        )

        self.save(rootdir, self.cfg.run_name, self.data_log)

        wandb.finish()
        self.training = False
        clear_output()
        return self.data_log

    def save(self, dir: pathlib.Path, run_name: str, dynamics_dict: dict):
        """ Saves model weights, projection vector, and training dynamics """
        weights_fname = dir/f'weights_{run_name}.pt'
        t.save(self.state_dict(), weights_fname)

        dynamics_fname = dir/f'dynamics_{run_name}.h5'
        with h5py.File(dynamics_fname, 'w') as f:
            for k, item in dynamics_dict.items():
                f[k] = np.array(item)

    def load(self, dir: pathlib.Path, run_name: str) -> dict:
        """ Loads model weights, projection vector, and training dynamics """
        weights_fname = dir/f'weights_{run_name}.pt'
        self.load_state_dict(t.load(weights_fname, map_location=device))

        dynamics_traces = {}
        dynamics_fname = dir/f'dynamics_{run_name}.h5'
        with h5py.File(dynamics_fname, 'r') as f:
            for k in ["step", "mse_loss", "l1_loss", "ghost_grad_loss", "overall_loss",
                      "explained_variance", "explained_variance_std", "l0", "strong_l0", "monosemanticity",
                      "mean_passes_since_fired", "n_passes_since_fired_over_threshold",
                      "below_1e-5", "below_1e-6", "dead_features", "n_training_tokens",
                      "current_learning_rate"]:
                dynamics_traces[k] = f[k][()]
        return dynamics_traces

    def __repr__(self):
        return f"AutoEncoder(d_mlp={self.cfg.d_mlp}, dict_mult={self.cfg.dict_mult})"
