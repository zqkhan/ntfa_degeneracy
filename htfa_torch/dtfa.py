"""Perform deep topographic factor analysis on fMRI data"""

__author__ = ('Jan-Willem van de Meent',
              'Eli Sennesh',
              'Zulqarnain Khan')
__email__ = ('j.vandemeent@northeastern.edu',
             'e.sennesh@northeastern.edu',
             'khan.zu@husky.neu.edu')

import collections
import datetime
import logging
import os
import os.path
import pickle
import time
import itertools

try:
    if __name__ == '__main__':
        import matplotlib
        matplotlib.use('TkAgg')
finally:
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
import nilearn.image
import nilearn.plotting as niplot
import numpy as np
from ordered_set import OrderedSet
import scipy.io as sio
import torch
import torch.distributions as dists
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.functional import softplus
import torch.optim.lr_scheduler

import probtorch

from . import dtfa_models
from . import tfa
from . import tfa_models
from . import utils

EPOCH_MSG = '[Epoch %d] (%dms) ELBO %.8e = log-likelihood %.8e - KL from prior %.8e, ' \
            'P weight penalty %.8e, S weight penalty %.8e, I weight penalty %.8e, Voxel Noise %.5e'

class DeepTFA:
    """Overall container for a run of Deep TFA

    ...

    Attributes
    ----------
    num_factors : int
        Number of spacial factors to be used
    num_blocks : int
        Number of blocks to use during training
    voxel_locations : torch.Tensor
        Flattened voxel array by (x,y,z) coordinates
    activation_normalizers : array_like of torch.Tensor 
        TODO
    activation_sufficient_stats : array_like
        TODO
    num_times : array_like of ints
        TODO
    num_voxels : int
        number of total voxels extracted from the dataset
    decoder : htfa_torch.dtfa_models.DeepTFADecoder
        Decorder model
    generative : htfa_torch.dtfa_models.DeepTFAModel
        Generative model
    variational : htfa_torch.dtfa_models.DeepTFAGuide
        Variational model
    optimizer : torch.optim.Optimizer
        Optimizer for all models
    scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
        Scheduler for all models
    _time_series : bool
        Time series flag to be passed on to model constructors
    _common_name : TODO
        TODO
    _dataset : htfa_torch.tardb.FmriTarDataset
        Tar dataset to train and evaluate on
    _subjects : array_like of ints
        Subject identifiers from dataset
    _tasks : array_like of Strings
        Stimuli labels from dataset

    Methods
    -------
    data(batch_size=None, selector=None)
        Prints the animals name and what sound it makes
    inference_filter
    """

    def __init__(self, data_tar, num_factors=tfa_models.NUM_FACTORS,
                 linear_params='', embedding_dim=2,
                 model_time_series=True, query_name=None, voxel_noise=tfa_models.VOXEL_NOISE
                ):
        """Example function with types documented in the docstring.

        `PEP 484`_ type annotations are supported. If attribute, parameter, and
        return types are annotated according to `PEP 484`_, they do not need to be
        included in the docstring:

        Parameters
        ----------
        param1 : int
            The first parameter.
        param2 : str
            The second parameter.

        """

        self.num_factors = num_factors
        self._time_series = model_time_series
        self._common_name = query_name
        self._dataset = data_tar
        self.num_blocks = len(self._dataset.blocks)

        self.voxel_locations = self._dataset.voxel_locations
        if tfa.CUDA:
            self.voxel_locations = self.voxel_locations.pin_memory()
        self._subjects = self._dataset.subjects()
        self._tasks = self._dataset.tasks()
        self._interactions = [x for x in itertools.product(self._subjects, self._tasks)]
        self.activation_normalizers, self.activation_sufficient_stats =\
            self._dataset.normalize_activations()

        # Pull out relevant dimensions: the number of time instants and the
        # number of voxels in each timewise "slice"
        self.num_times = [len(block['times']) for block
                          in self._dataset.blocks.values()]
        self.num_voxels = self.voxel_locations.shape[0]

        block_subjects = [self._subjects.index(b['subject'])
                          for b in self._dataset.blocks.values()]
        block_tasks = [self._tasks.index(b['task']) for b in
                       self._dataset.blocks.values()]
        block_interactions = [self._interactions.index((b['subject'], b['task']))
                              for b in self._dataset.blocks.values()]

        centers, widths, weights = utils.initial_hypermeans(
            self._dataset.mean_block().numpy().T, self.voxel_locations.numpy(),
            num_factors
        )
        hyper_means = {
            'weights': torch.Tensor(weights),
            'factor_centers': torch.Tensor(centers),
            'factor_log_widths': widths,
        }

        self.decoder = dtfa_models.DeepTFADecoder(self.num_factors,
                                                  self.voxel_locations,
                                                  embedding_dim,
                                                  time_series=model_time_series,
                                                  volume=True,
                                                  linear=linear_params)
        self.generative = dtfa_models.DeepTFAModel(
            self.voxel_locations, block_subjects, block_tasks, block_interactions,
            self.num_factors, self.num_blocks, self.num_times, embedding_dim, voxel_noise=voxel_noise,
        )
        self.variational = dtfa_models.DeepTFAGuide(self.num_factors,
                                                    block_subjects, block_tasks, block_interactions,
                                                    self.num_blocks,
                                                    self.num_times,
                                                    embedding_dim, hyper_means,
                                                    model_time_series)

        self.optimizer = None
        self.scheduler = None
        self._checkpoint_loaded = None
        self._inprogress = False


    def _init_optimizer_scheduler(self, learning_rate=tfa.LEARNING_RATE, train_globals=True, patience=10, param_tuning=False, learn_voxel_noise=False):
        if not isinstance(learning_rate, dict):
            learning_rate = {
                'q': learning_rate,
                'p': learning_rate / 10,
            }
            
        param_groups = [{
            'params': [phi for phi in self.variational.parameters()
                       if phi.shape[0] == self.num_blocks],
            'lr': learning_rate['q'],
        }, {
            'params': [theta for theta in self.decoder.parameters()
                       if theta.shape[0] == self.num_blocks],
            'lr': learning_rate['p'],
        }]

        if train_globals:
            param_groups.append({
                'params': [phi for phi in self.variational.parameters()
                           if phi.shape[0] != self.num_blocks],
                'lr': learning_rate['q'],
            })
            param_groups.append({
                'params': [theta for theta in self.decoder.parameters()
                           if theta.shape[0] != self.num_blocks],
                'lr': learning_rate['p'],
            })
        
        # if tuning, remove factor embedding parameters
        # loc 1 and 3 refer to decoder parameters indices above in param_groups
        if param_tuning:
            factor_params = [theta for theta in self.decoder.factors_embedding.parameters()]
            for i in range(len(factor_params)):
                for loc in [1,3]:
                    if factor_params[i] in param_groups[loc]['params']:
                        param_groups[loc]['params'].remove(factor_params[i])
        
        if learn_voxel_noise:
            self.generative.hyperparams.voxel_noise.requires_grad = True
            param_groups.append({
                'params': [self.generative.hyperparams.voxel_noise],
                'lr': learning_rate['p'],
            })

        self.optimizer = torch.optim.Adam(param_groups, amsgrad=True, eps=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, min_lr=1e-5, patience=patience,
            verbose=True
    )

    def subjects(self):
        return self._subjects

    def tasks(self):
        return self._tasks

    def num_parameters(self):
        parameters = list(self.variational.parameters()) +\
                     list(self.decoder.parameters())
        return sum([param.numel() for param in parameters])

    def start_times(self, blocks):
        starts = [self._dataset.blocks[block.item()]['times'][0] for block
                  in blocks]
        return torch.tensor(starts, dtype=torch.long, device=blocks.device)

    def relative_times(self, blocks, times):
        starts = self.start_times(blocks)
        return times - starts

    def train(self, num_steps=10, num_steps_exist=0, learning_rate=tfa.LEARNING_RATE,
              log_level=logging.WARNING, num_particles=tfa_models.NUM_PARTICLES, 
              batch_size=256, use_cuda=True, checkpoint_steps=None, patience=10,
              train_globals=True, blocks_filter=lambda block: True,
              l_p=0, l_s=0, l_i=0, param_tuning=False, learn_voxel_noise=False, path='./'):
        """Optimize the variational guide to reflect the data for `num_steps`"""
        logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=log_level)
        # S x T x V -> T x S x V
        training_data = torch.utils.data.DataLoader(
            self._dataset.data(selector=blocks_filter), batch_size=batch_size,
            pin_memory=True
        )
        decoder = self.decoder
        variational = self.variational
        generative = self.generative
        voxel_locations = self.voxel_locations

        if self.optimizer is None or self.scheduler is None:
            self._init_optimizer_scheduler(learning_rate, train_globals, patience, param_tuning, learn_voxel_noise)
        if self._checkpoint_loaded is not None and not self._inprogress:
            self.load_state_lr(self._checkpoint_loaded)

        optimizer = self.optimizer
        scheduler = self.scheduler

        self._inprogress = True

        if tfa.CUDA and use_cuda:
            decoder.cuda()
            variational.cuda()
            generative.cuda()
            voxel_locations = voxel_locations.cuda(non_blocking=True)
            self.optimizer_cuda()
            self.scheduler_cuda()


        decoder.train()
        variational.train()
        generative.train()

        free_energies = list(range(num_steps))
        p_w_penalties = list(range(num_steps))
        s_w_penalties = list(range(num_steps))
        i_w_penalties = list(range(num_steps))
        noise_param = list(range(num_steps))

        var_params = variational.hyperparams.state_vardict(num_particles)
        gen_params = generative.hyperparams.state_vardict(num_particles)

        for epoch in range(num_steps):
            start = time.time()
            epoch_free_energies = []
            epoch_lls = []
            epoch_prior_kls = []
            epoch_p_w_penalty = []
            epoch_s_w_penalty = []
            epoch_i_w_penalty = []

            for data in training_data:
                if tfa.CUDA and use_cuda:
                    data['activations'] = data['activations'].cuda(
                        non_blocking=True
                    )
                    data['block'] = data['block'].cuda(non_blocking=True)
                    data['t'] = data['t'].cuda(non_blocking=True)
                data['activations'] = data['activations'].expand(num_particles,
                                                                 -1, -1)
                rel_times = self.relative_times(data['block'], data['t'])

                optimizer.zero_grad()
                q = probtorch.Trace()
                variational(decoder, q, times=rel_times, blocks=data['block'],
                            params=var_params, num_particles=num_particles)
                p = probtorch.Trace()
                _, p_w, s_w, i_w = generative(decoder, p, times=rel_times, guide=q,
                           observations={'Y': data['activations']},
                           blocks=data['block'], locations=voxel_locations,
                           params=gen_params, num_particles=num_particles)
                p_w_norm = p_w.norm(p=1, dim=-1).sum()
                s_w_norm = s_w.norm(p=1, dim=-1).sum()
                i_w_norm = i_w.norm(p=1, dim=-1).sum()
                free_energy, ll, prior_kl = tfa.hierarchical_free_energy(
                    q, p,
                    num_particles=num_particles
                )

                penalized_free_energy = free_energy #+ l_p * p_w_norm + l_s * s_w_norm + l_i * i_w_norm

                penalized_free_energy.backward()
                optimizer.step()
                epoch_free_energies.append(penalized_free_energy.item())
                epoch_p_w_penalty.append(p_w_norm.item())
                epoch_s_w_penalty.append(s_w_norm.item())
                epoch_i_w_penalty.append(i_w_norm.item())

                epoch_lls.append(ll.item())
                epoch_prior_kls.append(prior_kl.item())

                if tfa.CUDA and use_cuda:
                    del data['activations']
                    del data['block']
                    del data['t']
                    torch.cuda.empty_cache()

            free_energies[epoch] = np.sum(epoch_free_energies)
            p_w_penalties[epoch] = np.sum(epoch_p_w_penalty)
            s_w_penalties[epoch] = np.sum(epoch_s_w_penalty)
            i_w_penalties[epoch] = np.sum(epoch_i_w_penalty)
            noise_param[epoch] = np.exp(self.generative.hyperparams.voxel_noise.item())

            scheduler.step(free_energies[epoch])

            end = time.time()
            # num_steps_exist accounts for prior epochs run if training
            # was started from an existing checkpoint using load_state()
            msg = EPOCH_MSG % (epoch + 1 + num_steps_exist, (end - start) * 1000,
                               -free_energies[epoch], np.sum(epoch_lls),
                               np.sum(epoch_prior_kls), np.sum(epoch_p_w_penalty),
                               np.sum(epoch_s_w_penalty), np.sum(epoch_i_w_penalty), noise_param[epoch])
            logging.info(msg)
            
            if (checkpoint_steps is not None and (epoch+1) % checkpoint_steps == 0) or \
               ((epoch+1) == num_steps):
                # save model checkpoint
                now = datetime.datetime.now()
                checkpoint_name = now.strftime(tfa.CHECKPOINT_TAG) + '_Epoch' + str(epoch + 1 + num_steps_exist)
                self.save_state(path=path, tag=checkpoint_name)
                # save losses at this checkpoint (since previous checkpoint)
                np.savetxt(path + self.common_name() + checkpoint_name + '_losses.txt',
                           free_energies[((epoch+1)-checkpoint_steps):(epoch+1)])
                logging.info('Saved checkpoint...')

        if tfa.CUDA and use_cuda:
            del voxel_locations
            decoder.cpu()
            variational.cpu()
            generative.cpu()

        return np.vstack((np.vstack([free_energies]) + np.vstack([p_w_penalties]) \
               + np.vstack([s_w_penalties]) + np.vstack([i_w_penalties]), noise_param))
    
                
    def free_energy(self, batch_size=64, use_cuda=True,
                    blocks_filter=lambda block: True, num_particles=1,
                    sample_size=1, predictive=False, ablate_subjects=False,
                    ablate_tasks=False, custom_interaction=None, custom_block=None):
        testing_data = torch.utils.data.DataLoader(
            self._dataset.data(selector=blocks_filter), batch_size=batch_size,
            pin_memory=True
        )
        log_likelihoods = torch.zeros(sample_size, len(testing_data))
        prior_kls = torch.zeros(sample_size, len(testing_data))
        self.decoder.eval()
        self.variational.eval()
        self.generative.eval()
        decoder = self.decoder
        variational = self.variational
        generative = self.generative
        voxel_locations = self.voxel_locations
        if custom_interaction is not None:
            custom_interaction = torch.tensor(custom_interaction)
        if tfa.CUDA and use_cuda:
            decoder.cuda()
            variational.cuda()
            generative.cuda()
            voxel_locations = voxel_locations.cuda().detach()
            log_likelihoods = log_likelihoods.to(voxel_locations)
            prior_kls = prior_kls.to(voxel_locations)
            if custom_interaction is not None:
                custom_interaction = custom_interaction.cuda()

        if sample_size > 1:
            for k in range(sample_size // num_particles):
                for (batch, data) in enumerate(testing_data):
                    if tfa.CUDA and use_cuda:
                        for key, val in data.items():
                            if isinstance(val, torch.Tensor):
                                data[key] = val.cuda()

                    if custom_block is not None:
                        original_block = data['block'].clone()
                        data['block'] = torch.ones_like(data['block']) * custom_block
                    else:
                        original_block = custom_block

                    rel_times = self.relative_times(data['block'], data['t'])

                    q = probtorch.Trace()
                    variational(decoder, q, times=rel_times, blocks=data['block'],
                                num_particles=num_particles,
                                ablate_subjects=ablate_subjects, ablate_tasks=ablate_tasks,
                                custom_interaction=custom_interaction, predictive=predictive,
                                block_subjects_factors=original_block)
                    p = probtorch.Trace()
                    generative(decoder, p, times=rel_times, guide=q,
                               observations={'Y': data['activations']},
                               blocks=data['block'], locations=voxel_locations,
                               num_particles=num_particles,
                               ablate_subjects=ablate_subjects, ablate_tasks=ablate_tasks,
                               custom_interaction=custom_interaction, block_subjects_factors=original_block)

                    _, ll, prior_kl = tfa.hierarchical_free_energy(
                        q, p, num_particles=num_particles
                    )

                    start = k * num_particles
                    end = (k + 1) * num_particles
                    log_likelihoods[start:end, batch] += ll.detach()
                    prior_kls[start:end, batch] += prior_kl.detach()

                    if tfa.CUDA and use_cuda:
                        for key, val in data.items():
                            if isinstance(val, torch.Tensor):
                                del val
                        torch.cuda.empty_cache()
        else:
            for (batch, data) in enumerate(testing_data):
                if tfa.CUDA and use_cuda:
                    for key, val in data.items():
                        if isinstance(val, torch.Tensor):
                            data[key] = val.cuda()

                if custom_block is not None:
                    original_block = data['block'].clone()
                    data['block'] = torch.ones_like(data['block']) * custom_block
                else:
                    original_block = custom_block

                rel_times = self.relative_times(data['block'], data['t'])

                q = probtorch.Trace()
                variational(decoder, q, times=rel_times, blocks=data['block'],
                            num_particles=num_particles,
                            ablate_subjects=ablate_subjects, ablate_tasks=ablate_tasks,
                            custom_interaction=custom_interaction, predictive=predictive,
                            block_subjects_factors=original_block, use_mean=True)
                p = probtorch.Trace()
                generative(decoder, p, times=rel_times, guide=q,
                           observations={'Y': data['activations']},
                           blocks=data['block'], locations=voxel_locations,
                           num_particles=num_particles,
                           ablate_subjects=ablate_subjects, ablate_tasks=ablate_tasks,
                           custom_interaction=custom_interaction, block_subjects_factors=original_block)

                _, ll, prior_kl = tfa.hierarchical_free_energy(
                    q, p, num_particles=num_particles
                )

                start = 0 * num_particles
                end = 1 * num_particles
                log_likelihoods[start:end, batch] += ll.detach()
                prior_kls[start:end, batch] += prior_kl.detach()

                if tfa.CUDA and use_cuda:
                    for key, val in data.items():
                        if isinstance(val, torch.Tensor):
                            del val
                    torch.cuda.empty_cache()

        if tfa.CUDA and use_cuda:
            del voxel_locations
            decoder.cpu()
            variational.cpu()
            generative.cpu()
            log_likelihoods = log_likelihoods.cpu()
            prior_kls = prior_kls.cpu()

        log_likelihood = log_likelihoods.sum(dim=-1)
        prior_kl = prior_kls.sum(dim=-1)
        elbo = log_likelihood - prior_kl
        iwae_log_likelihood = probtorch.util.log_mean_exp(log_likelihood,
                                                          dim=0).item()
        iwae_prior_kl = probtorch.util.log_mean_exp(prior_kl, dim=0).item()
        iwae_free_energy = probtorch.util.log_mean_exp(-elbo, dim=0).item()
        return [[-elbo.mean(dim=0).item(), log_likelihood.mean(dim=0).item(),
                 prior_kl.mean(dim=0).item()],
                [iwae_free_energy, iwae_log_likelihood, iwae_prior_kl]]

    def pred_log_like(self, use_cuda=True, testing_data=None, num_particles=1,
                    sample_size=1, predictive=False, ablate_subjects=False,
                    ablate_tasks=False, custom_interaction=None, custom_block=None):
        log_likelihoods = torch.zeros(sample_size, len(testing_data))
        prior_kls = torch.zeros(sample_size, len(testing_data))
        self.decoder.eval()
        self.variational.eval()
        self.generative.eval()
        decoder = self.decoder
        variational = self.variational
        generative = self.generative
        voxel_locations = self.voxel_locations
        if custom_interaction is not None:
            custom_interaction = torch.tensor(custom_interaction)
        if tfa.CUDA and use_cuda:
            decoder.cuda()
            variational.cuda()
            generative.cuda()
            voxel_locations = voxel_locations.cuda().detach()
            log_likelihoods = log_likelihoods.to(voxel_locations)
            prior_kls = prior_kls.to(voxel_locations)
            if custom_interaction is not None:
                custom_interaction = custom_interaction.cuda()

        if sample_size > 1:
            for k in range(sample_size // num_particles):
                for (batch, data) in enumerate(testing_data):
                    if tfa.CUDA and use_cuda:
                        for key, val in data.items():
                            if isinstance(val, torch.Tensor):
                                data[key] = val.cuda()

                    if custom_block is not None:
                        original_block = data['block'].clone()
                        data['block'] = torch.ones_like(data['block']) * custom_block
                    else:
                        original_block = custom_block

                    rel_times = self.relative_times(data['block'], data['t'])

                    q = probtorch.Trace()
                    variational(decoder, q, times=rel_times, blocks=data['block'],
                                num_particles=num_particles,
                                ablate_subjects=ablate_subjects, ablate_tasks=ablate_tasks,
                                custom_interaction=custom_interaction, predictive=predictive,
                                block_subjects_factors=original_block)
                    p = probtorch.Trace()
                    generative(decoder, p, times=rel_times, guide=q,
                               observations={'Y': data['activations']},
                               blocks=data['block'], locations=voxel_locations,
                               num_particles=num_particles,
                               ablate_subjects=ablate_subjects, ablate_tasks=ablate_tasks,
                               custom_interaction=custom_interaction, block_subjects_factors=original_block)

                    _, ll, prior_kl = tfa.hierarchical_free_energy(
                        q, p, num_particles=num_particles
                    )

                    start = k * num_particles
                    end = (k + 1) * num_particles
                    log_likelihoods[start:end, batch] += ll.detach()
                    prior_kls[start:end, batch] += prior_kl.detach()

                    if tfa.CUDA and use_cuda:
                        for key, val in data.items():
                            if isinstance(val, torch.Tensor):
                                del val
                        torch.cuda.empty_cache()
        else:
            for (batch, data) in enumerate(testing_data):
                if tfa.CUDA and use_cuda:
                    for key, val in data.items():
                        if isinstance(val, torch.Tensor):
                            data[key] = val.cuda()

                if custom_block is not None:
                    original_block = data['block'].clone()
                    data['block'] = torch.ones_like(data['block']) * custom_block
                else:
                    original_block = custom_block

                rel_times = self.relative_times(data['block'], data['t'])

                q = probtorch.Trace()
                variational(decoder, q, times=rel_times, blocks=data['block'],
                            num_particles=num_particles,
                            ablate_subjects=ablate_subjects, ablate_tasks=ablate_tasks,
                            custom_interaction=custom_interaction, predictive=predictive,
                            block_subjects_factors=original_block, use_mean=True)
                p = probtorch.Trace()
                generative(decoder, p, times=rel_times, guide=q,
                           observations={'Y': data['activations']},
                           blocks=data['block'], locations=voxel_locations,
                           num_particles=num_particles,
                           ablate_subjects=ablate_subjects, ablate_tasks=ablate_tasks,
                           custom_interaction=custom_interaction, block_subjects_factors=original_block)

                _, ll, prior_kl = tfa.hierarchical_free_energy(
                    q, p, num_particles=num_particles
                )

                start = 0 * num_particles
                end = 1 * num_particles
                log_likelihoods[start:end, batch] += ll.detach()
                prior_kls[start:end, batch] += prior_kl.detach()

                if tfa.CUDA and use_cuda:
                    for key, val in data.items():
                        if isinstance(val, torch.Tensor):
                            del val
                    torch.cuda.empty_cache()

        if tfa.CUDA and use_cuda:
            del voxel_locations
            decoder.cpu()
            variational.cpu()
            generative.cpu()
            log_likelihoods = log_likelihoods.cpu()
            prior_kls = prior_kls.cpu()

        return log_likelihoods

    def classification_matrix(self, validation_filter, save_file='classification.pk',
                              ablate_subjects=False, ablate_tasks=False,
                              custom_interaction=None, all_blocks=False, sample_size=1, use_cuda=True, row_numbers=None):
        block_subjects = [b['subject'] for b in self._dataset.blocks.values()]
        block_tasks = [b['task'] for b in self._dataset.blocks.values()]
        validation_blocks = [b for (b, block) in self._dataset.blocks.items() if validation_filter(block)]
        if row_numbers is None:
            validation_data = [self._dataset[block] for block in validation_blocks]
        else:
            validation_data = [self._dataset[block] for block in np.array(validation_blocks)[row_numbers]]
        log_likelihoods = torch.zeros(len(validation_blocks), len(validation_blocks))
        print("Starting")
        for (i_b, b) in (enumerate(validation_blocks)):
            print("Processing Block: " + str(b))
            log_likelihoods[row_numbers, i_b] = self.pred_log_like(use_cuda=use_cuda,
                                                         testing_data=validation_data,
                                                         sample_size=sample_size,
                                                         ablate_subjects=ablate_subjects,
                                                         ablate_tasks=ablate_tasks,
                                                         custom_interaction=custom_interaction,
                                                         predictive=True,
                                                         custom_block=b)
        classification_results = {'log_like': log_likelihoods,
                                  'soft_maxed': torch.nn.Softmax(dim=-1)(log_likelihoods),
                                  'validation_blocks': validation_blocks,
                                  'all_blocks': all_blocks,
                                  'validation_participants': np.array(block_subjects)[validation_blocks],
                                  'all_participants': np.array(block_subjects)[all_blocks],
                                  'validation_tasks': np.array(block_tasks)[validation_blocks],
                                  'all_tasks': np.array(block_tasks)[all_blocks]}
        pickle.dump(classification_results,open(save_file,'wb'))
        return classification_results

    def results(self, block=None, subject=None, task=None, interaction=None, times=None,
                hist_weights=False, generative=False,
                ablate_subjects=False, ablate_tasks=False, ablate_interactions=False):
        hyperparams = self.variational.hyperparams.state_vardict(1)

        guide = probtorch.Trace()
        if block is None:
            block = 0
        if times is None:
            times = torch.tensor(self._dataset.blocks[block]['times'],
                                 dtype=torch.long)
        subject = self._subjects.index(self._dataset.blocks[block]['subject'])
        task = self._tasks.index(self._dataset.blocks[block]['task'])
        interaction = self._interactions.index((self._dataset.blocks[block]['subject'],
                                                self._dataset.blocks[block]['task']))

        blocks = torch.tensor([block] * len(times), dtype=torch.long)
        subjects = torch.tensor([subject], dtype=torch.long)
        tasks = torch.tensor([task], dtype=torch.long)
        interactions = torch.tensor([interaction], dtype=torch.long)

        rel_times = self.relative_times(blocks, times)

        guide.variable(
            torch.distributions.Normal,
            hyperparams['subject']['mu'][:, subjects],
            torch.exp(hyperparams['subject']['log_sigma'][:, subjects]),
            value=hyperparams['subject']['mu'][:, subjects],
            name='z^PF',
        )
        if ablate_subjects:
            guide.variable(
                torch.distributions.Normal,
                hyperparams['subject_weight']['mu'][:, subjects],
                torch.exp(hyperparams['subject_weight']['log_sigma'][:, subjects]),
                value=torch.zeros_like(hyperparams['subject_weight']['mu'][:, subjects]),
                name='z^PW',
            )
        else:
            guide.variable(
                torch.distributions.Normal,
                hyperparams['subject_weight']['mu'][:, subjects],
                torch.exp(hyperparams['subject_weight']['log_sigma'][:, subjects]),
                value=hyperparams['subject_weight']['mu'][:, subjects],
                name='z^PW',
            )

        factor_centers_params = hyperparams['factor_centers']
        guide.variable(
            torch.distributions.Normal,
            factor_centers_params['mu'][:, subjects],
            torch.exp(factor_centers_params['log_sigma'][:, subjects]),
            value=factor_centers_params['mu'][:, subjects],
            name='FactorCenters',
        )
        factor_log_widths_params = hyperparams['factor_log_widths']
        guide.variable(
            torch.distributions.Normal,
            factor_log_widths_params['mu'][:, subjects],
            torch.exp(factor_log_widths_params['log_sigma'][:, subjects]),
            value=factor_log_widths_params['mu'][:, subjects],
            name='FactorLogWidths',
        )
        if ablate_tasks:
            guide.variable(
                torch.distributions.Normal,
                hyperparams['task']['mu'][:, tasks],
                torch.exp(hyperparams['task']['log_sigma'][:, tasks]),
                value=torch.zeros_like(hyperparams['task']['mu'][:, tasks]),
                name='z^S',
            )
        else:
            guide.variable(
                torch.distributions.Normal,
                hyperparams['task']['mu'][:, tasks],
                torch.exp(hyperparams['task']['log_sigma'][:, tasks]),
                value=hyperparams['task']['mu'][:, tasks],
                name='z^S',
            )

        if ablate_interactions:
            guide.variable(
                torch.distributions.Normal,
                hyperparams['interaction']['mu'][:, interactions],
                torch.exp(hyperparams['interaction']['log_sigma'][:, interactions]),
                value=torch.zeros_like(hyperparams['interaction']['mu'][:, interactions]),
                name='z^I',
            )
        else:
            guide.variable(
                torch.distributions.Normal,
                hyperparams['interaction']['mu'][:, interactions],
                torch.exp(hyperparams['interaction']['log_sigma'][:, interactions]),
                value=hyperparams['interaction']['mu'][:, interactions],
                name='z^I',
            )

        if self._time_series and not generative:
            weights_params = hyperparams['weights']
            guide.variable(
                torch.distributions.Normal,
                weights_params['mu'][:, blocks, rel_times],
                torch.exp(weights_params['log_sigma'][:, blocks, rel_times]),
                value=weights_params['mu'][:, blocks, rel_times],
                name='Weights_%s' % [t.item() for t in times]
            )

        weights, factor_centers, factor_log_widths, _,  _,  _ =\
            self.decoder(probtorch.Trace(), blocks, subjects, tasks, interactions,
                         hyperparams, rel_times, guide=guide,
                         generative=generative, ablate_subjects=ablate_subjects, ablate_tasks=ablate_tasks)

        weights = weights.squeeze(0)
        factor_centers = factor_centers[:, 0].squeeze(0)
        factor_log_widths = factor_log_widths[:, 0].squeeze(0)

        if hist_weights:
            plt.hist(weights.view(weights.numel()).data.numpy())
            plt.show()

        result = {
            'weights': weights[rel_times].data,
            'factors': tfa_models.radial_basis(self.voxel_locations,
                                               factor_centers.data,
                                               factor_log_widths.data),
            'factor_centers': factor_centers.data,
            'factor_log_widths': factor_log_widths.data,
        }
        if subject is not None:
            result['z^P'] = hyperparams['subject']['mu'][0, subject]
        if task is not None:
            result['z^S'] = hyperparams['task']['mu'][0, task]
        if interaction is not None:
            result['z^I'] = hyperparams['interaction']['mu'][0, interaction]

        return result

    def reconstruction(self, block=None, subject=None, task=None, interaction=None, t=0, ablate_subjects=False, ablate_tasks=False):
        results = self.results(block, subject, task, interaction, generative=t is None,
                               ablate_tasks=ablate_tasks, ablate_subjects=ablate_subjects)
        reconstruction = results['weights'] @ results['factors']

        image = utils.cmu2nii(reconstruction.numpy(),
                              self.voxel_locations.numpy(),
                              self._dataset.blocks[block]['template'])
        if t is None:
            image_slice = nilearn.image.mean_img(image)
            reconstruction = reconstruction.mean(dim=0, keepdim=True)
        else:
            image_slice = nilearn.image.index_img(image, t)
            reconstruction = reconstruction[t]
        return image_slice, reconstruction

    def reconstruction_diff(self, block, t=0, zscore_bound=3):
        activations = self._dataset[block]['activations']
        if t is None:
            activations = activations.mean(dim=0, keepdim=True)
        _, reconstruction = self.reconstruction(block, t=t)
        squared_diff = (activations - reconstruction) ** 2

        if zscore_bound is None:
            zscore_bound = squared_diff.max().item()

        image = utils.cmu2nii(squared_diff.numpy(),
                              self.voxel_locations.numpy(),
                              self._dataset.blocks[block]['template'])
        if t is None:
            image_slice = nilearn.image.mean_img(image)
        else:
            image_slice = nilearn.image.index_img(image, t)
            squared_diff = squared_diff[t]

        return image_slice, squared_diff

    def plot_reconstruction_diff(self, block=0, filename='', show=True, t=0,
                                 labeler=lambda b: None, zscore_bound=3,
                                 **kwargs):
        if filename == '' and t is None:
            filename = '%s-%s_ntfa_reconstruction_diff.pdf'
            filename = filename % (self.common_name(), str(block))
        elif filename == '':
            filename = '%s-%s_ntfa_reconstruction_diff_tr%d.pdf'
            filename = filename % (self.common_name(), str(block), t)

        image_slice, diff = self.reconstruction_diff(block, t=t,
                                                     zscore_bound=zscore_bound)
        plot = niplot.plot_glass_brain(
            image_slice, plot_abs=True, colorbar=True, symmetric_cbar=False,
            title=utils.title_brain_plot(block, self._dataset.blocks[block],
                                         labeler, t, 'Squared Residual'),
            vmin=0, vmax=zscore_bound ** 2, **kwargs,
        )

        if t is None:
            activations = self._dataset[block]['activations'].mean(dim=0,
                                                                   keepdim=True)
        else:
            activations = self._dataset[block]['activations'][t]

        logging.info(
            'Reconstruction Error (Frobenius Norm): %.8e out of %.8e',
            np.linalg.norm(diff.sqrt().numpy()),
            np.linalg.norm(activations.numpy())
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def plot_factor_centers(self, block, filename='', show=True, labeler=None,
                            serialize_data=True):
        if filename == '':
            filename = self.common_name() + '-' + str(block) +\
                       '_factor_centers.pdf'
        if labeler is None:
            labeler = lambda b: None
        results = self.results(block)

        centers_sizes = np.repeat([50], self.num_factors)
        sizes = torch.exp(results['factor_log_widths']).numpy()

        centers = results['factor_centers'].numpy()

        if serialize_data:
            tensors_filename = os.path.splitext(filename)[0] + '.dat'
            tensors = {
                'centers': torch.tensor(centers),
                'sizes': torch.tensor(sizes),
            }
            torch.save(tensors, tensors_filename)

        plot = niplot.plot_connectome(
            np.eye(self.num_factors * 2),
            np.vstack([centers, centers]),
            node_size=np.vstack([sizes, centers_sizes]),
            title=utils.title_brain_plot(block, self._dataset.blocks[block],
                                         labeler, None, 'Factor Centers'),
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def plot_original_brain(self, block=None, filename='', show=True,
                            plot_abs=False, t=0, labeler=None, zscore_bound=3,
                            **kwargs):
        if zscore_bound is None:
            zscore_bound = self.activation_normalizers[block]
        if filename == '' and t is None:
            filename = '%s-%s_original_brain.pdf' % (self.common_name(),
                                                     str(block))
        elif filename == '':
            filename = '%s-%s_original_brain_tr%d.pdf'
            filename = filename % (self.common_name(), str(block), t)
        if labeler is None:
            labeler = lambda b: None
        if block is None:
            block = np.random.choice(self.num_blocks, 1)[0]

        image = utils.cmu2nii(self._dataset[block]['activations'].numpy(),
                              self.voxel_locations.numpy(),
                              self._dataset.blocks[block]['template'])
        if t is None:
            image_slice = nilearn.image.mean_img(image)
        else:
            image_slice = nilearn.image.index_img(image, t)
        plot = niplot.plot_glass_brain(
            image_slice, plot_abs=plot_abs, colorbar=True, symmetric_cbar=True,
            title=utils.title_brain_plot(block, self._dataset.blocks[block],
                                         labeler, t),
            vmin=-zscore_bound, vmax=zscore_bound, **kwargs,
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def average_reconstruction_error(self, weighted=True,
                                     blocks_filter=lambda block: True):
        blocks = [block for block in range(self.num_blocks)
                  if blocks_filter(self._dataset.blocks[block])]


        if weighted:
            return utils.average_weighted_reconstruction_error(
                blocks, self.num_times, self.num_voxels,
                self._dataset, self.results
            )
        else:
            return utils.average_reconstruction_error(
                blocks, self._dataset, self.results
            )

    def plot_reconstruction(self, block=0, filename='', show=True,
                            plot_abs=False, t=0, labeler=None, zscore_bound=3,
                            ablate_subjects=False, ablate_tasks=False,
                            **kwargs):
        if zscore_bound is None:
            zscore_bound = self.activation_normalizers[block]
        if filename == '' and t is None:
            filename = '%s-%s_ntfa_reconstruction.pdf' % (self.common_name(),
                                                          str(block))
        elif filename == '':
            filename = '%s-%s_ntfa_reconstruction_tr%d.pdf'
            filename = filename % (self.common_name(), str(block), t)
        if labeler is None:
            labeler = lambda b: None

        image_slice, reconstruction = self.reconstruction(block=block, t=t,
                                                          ablate_subjects=ablate_subjects,
                                                          ablate_tasks=ablate_tasks)
        plot = niplot.plot_glass_brain(
            image_slice, plot_abs=plot_abs, colorbar=True, symmetric_cbar=True,
            title=utils.title_brain_plot(block, self._dataset.blocks[block],
                                         labeler, t, 'NeuralTFA'),
            vmin=-zscore_bound, vmax=zscore_bound, **kwargs,
        )

        activations = self._dataset[block]['activations']
        if t is None:
            activations = activations.mean(dim=0, keepdim=True)
        else:
            activations = activations[t]

        logging.info(
            'Reconstruction Error (Frobenius Norm): %.8e out of %.8e',
            np.linalg.norm((activations - reconstruction).numpy()),
            np.linalg.norm(activations.numpy())
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def plot_subject_template(self, subject, filename='', show=True,
                              plot_abs=False, serialize_data=True,
                              zscore_bound=3, **kwargs):
        if filename == '':
            filename = self.common_name() + '-' + str(subject) +\
                       '_subject_template.pdf'
        i = self.subjects().index(subject)
        results = self.results(block=None, task=None, subject=i)
        template = [i for (i, b) in enumerate(self._dataset.blocks.values())
                    if b['subject'] == subject][0]
        reconstruction = results['weights'] @ results['factors']
        if zscore_bound is None:
            zscore_bound = self.activation_normalizers[template]

        image = utils.cmu2nii(reconstruction.numpy(),
                              self.voxel_locations.numpy(),
                              self._dataset.blocks[template]['template'])
        image_slice = nilearn.image.index_img(image, 0)

        if serialize_data:
            tensors_filename = os.path.splitext(filename)[0] + '.dat'
            tensors = {
                'reconstruction': reconstruction,
                'voxel_locations': self.voxel_locations,
                'template': self._dataset.blocks[template]['template'],
                'activation_normalizer': self.activation_normalizers[template],
            }
            torch.save(tensors, tensors_filename)

        plot = niplot.plot_glass_brain(
            image_slice, plot_abs=plot_abs, colorbar=True, symmetric_cbar=True,
            title="Template for Participant %d" % subject,
            vmin=-zscore_bound, vmax=zscore_bound, **kwargs,
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def plot_task_template(self, task, filename='', show=True, plot_abs=False,
                           labeler=lambda x: x, serialize_data=True,
                           zscore_bound=3, **kwargs):
        if filename == '':
            filename = self.common_name() + '-' + str(task) +\
                       '_task_template.pdf'
        i = self.tasks().index(task)
        results = self.results(block=None, subject=None, task=i)
        template = [i for (i, b) in enumerate(self._dataset.blocks.values())
                    if b['task'] == task][0]
        reconstruction = results['weights'] @ results['factors']
        if zscore_bound is None:
            zscore_bound = self.activation_normalizers[template]

        image = utils.cmu2nii(reconstruction.numpy(),
                              self.voxel_locations.numpy(),
                              self._dataset.blocks[template]['template'])
        image_slice = nilearn.image.index_img(image, 0)

        if serialize_data:
            tensors_filename = os.path.splitext(filename)[0] + '.dat'
            tensors = {
                'reconstruction': reconstruction,
                'voxel_locations': self.voxel_locations,
                'template': self._dataset.blocks[template]['template'],
                'activation_normalizer': self.activation_normalizers[template],
            }
            torch.save(tensors, tensors_filename)

        plot = niplot.plot_glass_brain(
            image_slice, plot_abs=plot_abs, colorbar=True, symmetric_cbar=True,
            title="Template for Stimulus '%s'" % labeler(task),
            vmin=-zscore_bound, vmax=zscore_bound, **kwargs,
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def visualize_factor_embedding(self, filename='', show=True,
                                   hist_log_widths=True, serialize_data=True,
                                   **kwargs):
        if filename == '':
            filename = self.common_name() + '_factor_embedding.pdf'
        results = self.results(block=None, subject=None, task=None)
        centers = results['factor_centers']
        log_widths = results['factor_log_widths']
        widths = torch.exp(log_widths)

        if serialize_data:
            tensors_filename = os.path.splitext(filename)[0] + '.dat'
            tensors = {
                'centers': centers,
                'widths': widths,
                'num_factors': self.num_factors
            }
            torch.save(tensors, tensors_filename)

        plot = niplot.plot_connectome(
            np.eye(self.num_factors),
            centers.view(self.num_factors, 3).numpy(),
            node_size=widths.view(self.num_factors).numpy(),
            title="$x^F$ std-dev %.8e, $\\rho^F$ std-dev %.8e" %
            (centers.std(0).norm(), log_widths.std(0).norm()),
            **kwargs
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        if hist_log_widths:
            plt.hist(log_widths.view(log_widths.numel()).numpy())
            plt.show()

        return plot, centers, log_widths

    def heatmap_subject_embedding(self, heatmaps=[], filename='', show=True,
                                  xlims=None, ylims=None, figsize=utils.FIGSIZE,
                                  colormap=plt.rcParams['image.cmap'],
                                  serialize_data=True, plot_ellipse=True,
                                  legend_ordering=None, titles=[]):
        if filename == '':
            filename = self.common_name() + '_subject_heatmap.pdf'
        hyperparams = self.variational.hyperparams.state_vardict()
        z_p_mu = hyperparams['subject']['mu'].data
        z_p_sigma = torch.exp(hyperparams['subject']['log_sigma'].data)
        subjects = self.subjects()

        minus_lims = torch.min(z_p_mu - z_p_sigma * 2, dim=0)[0].tolist()
        plus_lims = torch.max(z_p_mu + z_p_sigma * 2, dim=0)[0].tolist()
        if not xlims:
            xlims = (minus_lims[0], plus_lims[0])
        if not ylims:
            ylims = (minus_lims[1], plus_lims[1])

        if not heatmaps:
            heatmaps = [lambda s: 1.0]
        heats = [sorted([heatmap(s) for s in subjects]) for heatmap in heatmaps]

        if serialize_data:
            tensors_filename = os.path.splitext(filename)[0] + '.dat'
            tensors = {
                'z_p': {'mu': z_p_mu, 'sigma': z_p_sigma},
                'colormap': colormap,
                'z_heats': heats,
            }
            torch.save(tensors, tensors_filename)

        with plt.style.context('seaborn-white'):
            ncols = len(heatmaps)
            if figsize is not None:
                (w, h) = figsize
                figsize = (w * ncols, h)

            fig, axs = plt.subplots(nrows=1, ncols=ncols, facecolor='white',
                                    sharey=True, figsize=figsize, frameon=True)
            for c in range(ncols):
                palette = cm.ScalarMappable(None, colormap)
                subject_colors = palette.to_rgba(np.array(heats[c]), norm=True)
                palette.set_array(np.array(heats[c]))

                utils.plot_embedding_clusters(z_p_mu, z_p_sigma, subject_colors,
                                              '', titles[c], palette, axs[c],
                                              xlims=xlims, ylims=ylims,
                                              plot_ellipse=plot_ellipse,
                                              legend_ordering=legend_ordering,
                                              color_legend=False)

            fig.text(0.435, 0.05, '$z^P_1$', ha='center', va='center')
            fig.text(0.1, 0.5, '$z^P_2$', ha='center', va='center',
                     rotation='vertical')
            palette.set_clim(0., 1.)
            plt.colorbar(palette, ax=axs)

            if filename is not None:
                fig.savefig(filename)
            if show:
                fig.show()

    def scatter_subject_embedding(self, labeler=None, filename='', show=True,
                                  xlims=None, ylims=None, figsize=utils.FIGSIZE,
                                  colormap=plt.rcParams['image.cmap'],
                                  serialize_data=True, plot_ellipse=True,
                                  legend_ordering=None):
        if filename == '':
            filename = self.common_name() + '_subject_embedding.pdf'
        hyperparams = self.variational.hyperparams.state_vardict()
        z_p_mu = hyperparams['subject']['mu'].data
        z_p_sigma = torch.exp(hyperparams['subject']['log_sigma'].data)
        subjects = self.subjects()

        minus_lims = torch.min(z_p_mu - z_p_sigma * 2, dim=0)[0].tolist()
        plus_lims = torch.max(z_p_mu + z_p_sigma * 2, dim=0)[0].tolist()
        if not xlims:
            xlims = (minus_lims[0], plus_lims[0])
        if not ylims:
            ylims = (minus_lims[1], plus_lims[1])

        if labeler is None:
            labeler = lambda s: s
        labels = sorted(list({labeler(s) for s in subjects}))
        if all([isinstance(label, float) for label in labels]):
            palette = cm.ScalarMappable(None, colormap)
            subject_colors = palette.to_rgba(np.array(labels), norm=True)
            palette.set_array(np.array(labels))
        else:
            palette = dict(zip(
                labels, utils.compose_palette(len(labels), colormap=colormap)
            ))
            subject_colors = [palette[labeler(subject)] for subject in subjects]

        if serialize_data:
            tensors_filename = os.path.splitext(filename)[0] + '.dat'
            tensors = {
                'z_p': {'mu': z_p_mu, 'sigma': z_p_sigma},
                'palette': palette,
                'subject_colors': subject_colors,
                'labels': labels,
            }
            torch.save(tensors, tensors_filename)

        utils.embedding_clusters_fig(z_p_mu, z_p_sigma, subject_colors, 'z^P',
                                     'Participant Embeddings', palette,
                                     filename=filename, show=show, xlims=xlims,
                                     ylims=ylims, figsize=figsize,
                                     plot_ellipse=plot_ellipse,
                                     legend_ordering=legend_ordering)

    def scatter_subject_weight_embedding(self, labeler=None, filename='', show=True,
                                  xlims=None, ylims=None, figsize=utils.FIGSIZE,
                                  colormap=plt.rcParams['image.cmap'],
                                  serialize_data=True, plot_ellipse=True,
                                  legend_ordering=None):
        if filename == '':
            filename = self.common_name() + '_subject_weight_embedding.pdf'
        hyperparams = self.variational.hyperparams.state_vardict()
        z_p_mu = hyperparams['subject_weight']['mu'].data
        z_p_sigma = torch.exp(hyperparams['subject_weight']['log_sigma'].data)
        subjects = self.subjects()

        minus_lims = torch.min(z_p_mu - z_p_sigma * 2, dim=0)[0].tolist()
        plus_lims = torch.max(z_p_mu + z_p_sigma * 2, dim=0)[0].tolist()
        if not xlims:
            xlims = (minus_lims[0], plus_lims[0])
        if not ylims:
            ylims = (minus_lims[1], plus_lims[1])

        if labeler is None:
            labeler = lambda s: s
        labels = sorted(list({labeler(s) for s in subjects}))
        if all([isinstance(label, float) for label in labels]):
            palette = cm.ScalarMappable(None, colormap)
            subject_colors = palette.to_rgba(np.array(labels), norm=True)
            palette.set_array(np.array(labels))
        else:
            palette = dict(zip(
                labels, utils.compose_palette(len(labels), colormap=colormap)
            ))
            subject_colors = [palette[labeler(subject)] for subject in subjects]

        if serialize_data:
            tensors_filename = os.path.splitext(filename)[0] + '.dat'
            tensors = {
                'z_p': {'mu': z_p_mu, 'sigma': z_p_sigma},
                'palette': palette,
                'subject_colors': subject_colors,
                'labels': labels,
            }
            torch.save(tensors, tensors_filename)

        utils.embedding_clusters_fig(z_p_mu, z_p_sigma, subject_colors, 'z^P',
                                     'Participant Embeddings', palette,
                                     filename=filename, show=show, xlims=xlims,
                                     ylims=ylims, figsize=figsize,
                                     plot_ellipse=plot_ellipse,
                                     legend_ordering=legend_ordering)


    def scatter_task_embedding(self, labeler=None, filename='', show=True,
                               xlims=None, ylims=None, figsize=utils.FIGSIZE,
                               colormap=plt.rcParams['image.cmap'],
                               serialize_data=True, plot_ellipse=True,
                               legend_ordering=None):
        if filename == '':
            filename = self.common_name() + '_task_embedding.pdf'
        hyperparams = self.variational.hyperparams.state_vardict()
        z_s_mu = hyperparams['task']['mu'].data
        z_s_sigma = torch.exp(hyperparams['task']['log_sigma'].data)
        tasks = self.tasks()

        minus_lims = torch.min(z_s_mu - z_s_sigma * 2, dim=0)[0].tolist()
        plus_lims = torch.max(z_s_mu + z_s_sigma * 2, dim=0)[0].tolist()
        if not xlims:
            xlims = (minus_lims[0], plus_lims[0])
        if not ylims:
            ylims = (minus_lims[1], plus_lims[1])

        if labeler is None:
            labeler = lambda t: t
        labels = sorted(list({labeler(t) for t in tasks}))
        if all([isinstance(label, float) for label in labels]):
            palette = cm.ScalarMappable(None, colormap)
            task_colors = palette.to_rgba(np.array(labels), norm=True)
            palette.set_array(np.array(labels))
        else:
            palette = dict(zip(
                labels, utils.compose_palette(len(labels), colormap=colormap)
            ))
            task_colors = [palette[labeler(task)] for task in tasks]

        if serialize_data:
            tensors_filename = os.path.splitext(filename)[0] + '.dat'
            tensors = {
                'z_s': {'mu': z_s_mu, 'sigma': z_s_sigma},
                'palette': palette,
                'task_colors': task_colors,
                'labels': labels,
            }
            torch.save(tensors, tensors_filename)

        utils.embedding_clusters_fig(z_s_mu, z_s_sigma, task_colors, 'z^S',
                                     'Stimulus Embeddings', palette,
                                     filename=filename, show=show, xlims=xlims,
                                     ylims=ylims, figsize=figsize,
                                     plot_ellipse=plot_ellipse,
                                     legend_ordering=legend_ordering)

    def common_name(self):
        if not self._common_name:
            self._common_name = os.path.commonprefix(
                [os.path.basename(b['template']) for b
                 in self._dataset.blocks.values()]
            )
        return self._common_name

    def save_state(self, path='./', tag=''):
        name = self.common_name() + tag
        variational_state = self.variational.state_dict()
        torch.save(variational_state,
                   path + name + '.dtfa_guide')
        torch.save(self.decoder.state_dict(),
                   path + name + '.dtfa_model')
        torch.save(self.generative.state_dict(),
                   path + name + '.dtfa_generative')
        torch.save(self.scheduler.state_dict(),
                   path + name + '.dtfa_scheduler')
        torch.save(self.optimizer.state_dict(),
                   path + name + '.dtfa_optimizer')
                   
    def save(self, path='./'):
        name = self.common_name()
        torch.save(self.variational.state_dict(),
                   path + name + '.dtfa_guide')
        torch.save(self.decoder.state_dict(),
                   path + name + '.dtfa_model')
        torch.save(self.generative.state_dict(),
                   path + name + '.dtfa_generative')
        torch.save(self.scheduler.state_dict(),
                   path + name + '.dtfa_scheduler')
        torch.save(self.optimizer.state_dict(),
                   path + name + '.dtfa_optimizer')
        with open(path  + name + '.dtfa', 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    def load_state(self, basename, load_generative=True):
        """
        load_generative: Set to 'True' to load learned generative parameters e.g. voxel_noise
        """
        model_state = torch.load(basename + '.dtfa_model')
        self.decoder.load_state_dict(model_state)

        guide_state = torch.load(basename + '.dtfa_guide')
        self.variational.load_state_dict(guide_state)

        if load_generative:
            generative_state = torch.load(basename + '.dtfa_generative')
            self.generative.load_state_dict(generative_state)

        self._checkpoint_loaded = basename
        self._inprogress = False


    def load_state_lr(self, basename):

        optimizer_state = torch.load(basename + '.dtfa_optimizer')
        self.optimizer.load_state_dict(optimizer_state)

        scheduler_state = torch.load(basename + '.dtfa_scheduler')
        self.scheduler.load_state_dict(scheduler_state)

    @classmethod
    def load(cls, basename):
        with open(basename + '.dtfa', 'rb') as pickle_file:
            dtfa = pickle.load(pickle_file)
        dtfa.load_state(basename)

        return dtfa

    def optimizer_cuda(self):
        device = torch.device(str("cuda:0"))
        for param in self.optimizer.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

    def scheduler_cuda(self):
        device = torch.device(str("cuda:0"))
        for param in self.scheduler.__dict__.values():
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)


    def decoding_accuracy(self, labeler=lambda x: x, window_size=60):
        """
        :return: accuracy: a dict containing decoding accuracies for each task [activity,isfc,mixed]
        """
        tasks = np.unique([labeler(b.task) for b in self._dataset.blocks])
        group = {task: [] for task in tasks}
        accuracy = {task: {'node': [], 'isfc': [], 'mixed': [], 'kl': []}
                    for task in tasks}

        for (b, block) in self._dataset.blocks.items():
            factorization = self.results(b)
            group[(block['task'])].append(factorization['weights'])

        for task in set(tasks):
            print(task)
            group[task] = torch.stack(group[task])
            if group[task].shape[0] < 2:
                raise ValueError('Not enough subjects for task %s' % task)
            group1 = group[task][:group[task].shape[0] // 2]
            group2 = group[task][group[task].shape[0] // 2:]
            node_accuracy, node_correlation = utils.get_decoding_accuracy(
                group1.data.numpy(), group2.data.numpy(), window_size
            )
            accuracy[task]['node'].append(node_accuracy)
            isfc_accuracy, isfc_correlation = utils.get_isfc_decoding_accuracy(
                group1.data.numpy(), group2.data.numpy(), window_size
            )
            accuracy[task]['isfc'].append(isfc_accuracy)
            accuracy[task]['mixed'].append(
                utils.get_mixed_decoding_accuracy(node_correlation,
                                                  isfc_correlation)
            )
            accuracy[task]['kl'].append(
                utils.get_kl_decoding_accuracy(group1.data.numpy(),
                                               group2.data.numpy(), window_size)
            )

        return accuracy

    def voxel_decoding_accuracy(self, labeler=lambda x: x, window_size=60):
        times = self.num_times
        keys = np.unique([labeler(b.task) for b in self._dataset.blocks])
        group = {key: [] for key in keys}
        accuracy = {key: [] for key in keys}
        for key in keys:
            print(key)
            for n in range(self.num_blocks):
                if key == self._dataset.blocks[n]['task']:
                    group[key].append(self._dataset[n]['activations'][:times[n], :])
            group[key] = np.rollaxis(np.dstack(group[key]), -1)
            if group[key].shape[0] < 2:
                raise ValueError('not enough subjects for the task: ' + key)
            else:
                G1 = group[key][:int(group[key].shape[0] / 2), :, :]
                G2 = group[key][int(group[key].shape[0] / 2):, :, :]
                accuracy[key].append(
                    utils.get_decoding_accuracy(G1, G2, window_size)[0]
                )
        return accuracy
