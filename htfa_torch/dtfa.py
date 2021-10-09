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
import itertools
import probtorch

from . import dtfa_models
from . import tfa
from . import tfa_models
from . import utils

EPOCH_MSG = '[Epoch %d] (%dms) Posterior free-energy %.8e = KL from prior %.8e - log-likelihood %.8e'

class DeepTFA:
    """Overall container for a run of Deep TFA"""
    def __init__(self, query, mask, num_factors=tfa_models.NUM_FACTORS,
                 embedding_dim=2, model_time_series=True, query_name=None):
        self.num_factors = num_factors
        self._time_series = model_time_series
        self._common_name = query_name
        self.mask = mask
        self._blocks = list(query)
        for block in self._blocks:
            block.load()
            block.unload_locations()
        self.num_blocks = len(self._blocks)
        self.voxel_activations = [block.activations for block in self._blocks]
        self._blocks[-1].load()
        if tfa.CUDA:
            self.voxel_locations = self._blocks[-1].locations.pin_memory()
        else:
            self.voxel_locations = self._blocks[-1].locations
        self._templates = [block.filename for block in self._blocks]
        self._tasks = [block.task for block in self._blocks]

        self.activation_normalizers = None
        self.activation_sufficient_stats = None
        self.normalize_activations()

        # Pull out relevant dimensions: the number of time instants and the
        # number of voxels in each timewise "slice"
        self.num_times = [acts.shape[0] for acts in self.voxel_activations]
        self.num_voxels = self.voxel_locations.shape[0]

        subjects = self.subjects()
        tasks = self.tasks()
        self.num_tasks = len(tasks)
        # stimuli = self.stimuli()
        interactions = OrderedSet(list(itertools.product(subjects,tasks)))
        block_subjects = [subjects.index(b.subject) for b in self._blocks]
        block_tasks = [tasks.index(b.task) for b in self._blocks]
        # block_stimuli = [stimuli.index(b.individual_differences['stimulus']) for b in self._blocks]
        block_interactions = [interactions.index((b.subject,b.task))
                              for b in self._blocks]
        b = max(range(self.num_blocks), key=lambda b: self.num_times[b])
        init_activations = self.voxel_activations.copy()
        max_times = max(self.num_times)
        for i, acts in enumerate(init_activations):
            if acts.shape[0] < max_times:
                buffer = torch.zeros(max_times - acts.shape[0], self.num_voxels)
                init_activations[i] = torch.cat((acts, buffer))
        init_activations = torch.stack(init_activations)
        centers, widths, weights = utils.initial_hypermeans(
            init_activations.mean(dim=0).numpy().T, self.voxel_locations.numpy(),
            num_factors
        )
        hyper_means = {
            'weights': torch.Tensor(weights),
            'factor_centers': torch.Tensor(centers),
            'factor_log_widths': widths,
        }

        self.decoder = dtfa_models.DeepTFADecoder(self.num_factors, hyper_means,
                                                  self.num_tasks,
                                                  embedding_dim,
                                                  time_series=model_time_series)
        self.generative = dtfa_models.DeepTFAModel(
            self.voxel_locations, block_subjects, block_tasks , block_interactions,
            self.num_factors, self.num_blocks, self.num_times, embedding_dim
        )
        self.variational = dtfa_models.DeepTFAGuide(self.num_factors,
                                                    block_subjects, block_tasks,
                                                    block_interactions,
                                                    self.num_blocks,
                                                    self.num_times,
                                                    embedding_dim, hyper_means,
                                                    model_time_series)

    def subjects(self):
        return OrderedSet([b.subject for b in self._blocks])

    def tasks(self):
        return OrderedSet([b.task for b in self._blocks])

    def num_parameters(self):
        parameters = list(self.variational.parameters()) +\
                     list(self.decoder.parameters())
        return sum([param.numel() for param in parameters])

    def train(self, num_steps=10, learning_rate=tfa.LEARNING_RATE,
              log_level=logging.WARNING, num_particles=tfa_models.NUM_PARTICLES,
              batch_size=64, use_cuda=True, checkpoint_steps=None,
              blocks_batch_size=4, patience=10, train_globals=True,
              blocks_filter=lambda block: True):
        """Optimize the variational guide to reflect the data for `num_steps`"""
        logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=log_level)
        # S x T x V -> T x S x V
        training_blocks = [(b, block) for (b, block) in enumerate(self._blocks)
                           if blocks_filter(block)]
        activations_loader = torch.utils.data.DataLoader(
            utils.TFADataset([block.activations
                              for (_, block) in training_blocks]),
            batch_size=batch_size,
            pin_memory=True,
        )
        decoder = self.decoder
        variational = self.variational
        generative = self.generative
        voxel_locations = self.voxel_locations
        if tfa.CUDA and use_cuda:
            decoder.cuda()
            variational.cuda()
            generative.cuda()
            voxel_locations = voxel_locations.cuda()
        if not isinstance(learning_rate, dict):
            learning_rate = {
                'q': learning_rate,
                'p': learning_rate / 10,
            }

        param_groups = [{
            'params': [phi for phi in variational.parameters()
                       if phi.shape[0] == len(self._blocks)],
            'lr': learning_rate['q'],
        }, {
            'params': [theta for theta in decoder.parameters()
                       if theta.shape[0] == len(self._blocks)],
            'lr': learning_rate['p'],
        }]
        if train_globals:
            param_groups.append({
                'params': [phi for phi in variational.parameters()
                           if phi.shape[0] != len(self._blocks)],
                'lr': learning_rate['q'],
            })
            param_groups.append({
                'params': [theta for theta in decoder.parameters()
                           if theta.shape[0] != len(self._blocks)],
                'lr': learning_rate['p'],
            })
        optimizer = torch.optim.Adam(param_groups, amsgrad=True, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, min_lr=1e-5, patience=patience,
            verbose=True
        )
        variational.train()
        generative.train()

        free_energies = list(range(num_steps))
        rv_occurrences = collections.defaultdict(int)
        measure_occurrences = True

        for epoch in range(num_steps):
            start = time.time()
            epoch_free_energies = list(range(len(activations_loader)))
            epoch_lls = list(range(len(activations_loader)))
            epoch_prior_kls = list(range(len(activations_loader)))

            for (batch, data) in enumerate(activations_loader):
                epoch_free_energies[batch] = 0.0
                epoch_lls[batch] = 0.0
                epoch_prior_kls[batch] = 0.0
                block_batches = utils.chunks(list(range(len(training_blocks))),
                                             n=blocks_batch_size)
                for block_batch in block_batches:
                    activations = [{'Y': data[:, b, :]} for b in block_batch]
                    block_batch = [training_blocks[b][0] for b in block_batch]
                    if tfa.CUDA and use_cuda:
                        for acts in activations:
                            acts['Y'] = acts['Y'].cuda()
                    trs = (batch * batch_size, None)
                    trs = (trs[0], trs[0] + activations[0]['Y'].shape[0])

                    optimizer.zero_grad()
                    q = probtorch.Trace()
                    variational(decoder, q, times=trs, blocks=block_batch,
                                num_particles=num_particles)
                    p = probtorch.Trace()
                    generative(decoder, p, times=trs, guide=q,
                               observations=activations, blocks=block_batch,
                               locations=voxel_locations)

                    def block_rv_weight(node, prior=True):
                        result = 1.0
                        if measure_occurrences:
                            rv_occurrences[node] += 1
                        result /= rv_occurrences[node]
                        return result
                    free_energy, ll, prior_kl = tfa.hierarchical_free_energy(
                        q, p,
                        rv_weight=block_rv_weight,
                        num_particles=num_particles
                    )

                    free_energy.backward()
                    optimizer.step()
                    epoch_free_energies[batch] += free_energy
                    epoch_lls[batch] += ll
                    epoch_prior_kls[batch] += prior_kl

                    if tfa.CUDA and use_cuda:
                        del activations
                        torch.cuda.empty_cache()
                if tfa.CUDA and use_cuda:
                    epoch_free_energies[batch] = epoch_free_energies[batch].cpu().data.numpy()
                    epoch_lls[batch] = epoch_lls[batch].cpu().data.numpy()
                    epoch_prior_kls[batch] = epoch_prior_kls[batch].cpu().data.numpy()
                else:
                    epoch_free_energies[batch] = epoch_free_energies[batch].data.numpy()
                    epoch_lls[batch] = epoch_lls[batch].data.numpy()
                    epoch_prior_kls[batch] = epoch_prior_kls[batch].data.numpy()

            free_energies[epoch] = np.array(epoch_free_energies).mean(0)
            scheduler.step(free_energies[epoch])

            measure_occurrences = False

            end = time.time()
            msg = EPOCH_MSG % (epoch + 1, (end - start) * 1000,
                               free_energies[epoch], np.mean(epoch_prior_kls),
                               np.mean(epoch_lls))
            logging.info(msg)
            if checkpoint_steps is not None and epoch % checkpoint_steps == 0:
                now = datetime.datetime.now()
                checkpoint_name = now.strftime(tfa.CHECKPOINT_TAG)
                logging.info('Saving checkpoint...')
                self.save_state(path='.', tag=checkpoint_name)

        if tfa.CUDA and use_cuda:
            decoder.cpu()
            variational.cpu()
            generative.cpu()

        now = datetime.datetime.now()
        checkpoint_name = now.strftime(tfa.CHECKPOINT_TAG)
        logging.info('Saving checkpoint...')
        self.save_state(path='.', tag=checkpoint_name)

        return np.vstack([free_energies])

    def free_energy(self, batch_size=64, use_cuda=True, blocks_batch_size=4,
                    blocks_filter=lambda block: True, num_particles=1,
                    sample_size=10, predictive=False):
        testing_blocks = [(b, block) for (b, block) in enumerate(self._blocks)
                          if blocks_filter(block)]
        activations_loader = torch.utils.data.DataLoader(
            utils.TFADataset([block.activations.detach()
                              for (_, block) in testing_blocks]),
            batch_size=batch_size,
            pin_memory=True,
        )
        log_likelihoods = torch.zeros(sample_size, len(activations_loader))
        prior_kls = torch.zeros(sample_size, len(activations_loader))

        self.decoder.eval()
        self.variational.eval()
        self.generative.eval()
        decoder = self.decoder
        variational = self.variational
        generative = self.generative
        voxel_locations = self.voxel_locations
        if tfa.CUDA and use_cuda:
            decoder.cuda()
            variational.cuda()
            generative.cuda()
            voxel_locations = voxel_locations.cuda().detach()
            log_likelihoods = log_likelihoods.to(voxel_locations)
            prior_kls = prior_kls.to(voxel_locations)

        for k in range(sample_size // num_particles):
            for (batch, data) in enumerate(activations_loader):
                block_batches = utils.chunks(list(range(len(testing_blocks))),
                                             n=blocks_batch_size)
                for block_batch in block_batches:
                    activations = [{'Y': data[:, b, :]} for b in block_batch]
                    block_batch = [testing_blocks[b][0] for b in block_batch]
                    if tfa.CUDA and use_cuda:
                        for acts in activations:
                            acts['Y'] = acts['Y'].cuda()
                    trs = (batch * batch_size, None)
                    trs = (trs[0], trs[0] + activations[0]['Y'].shape[0])

                    q = probtorch.Trace()
                    variational(decoder, q, times=trs, blocks=block_batch,
                                num_particles=num_particles)
                    p = probtorch.Trace()
                    generative(decoder, p, times=trs, guide=q,
                               observations=activations, blocks=block_batch,
                               locations=voxel_locations)

                    _, ll, prior_kl = tfa.hierarchical_free_energy(
                        q, p, num_particles=num_particles
                    )

                    start = k * num_particles
                    end = (k + 1) * num_particles
                    log_likelihoods[start:end, batch] += ll.detach()
                    prior_kls[start:end, batch] += prior_kl.detach()

                    if tfa.CUDA and use_cuda:
                        del activations
                        torch.cuda.empty_cache()

        if tfa.CUDA and use_cuda:
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

    def results(self, block=None, subject=None, task=None, interaction=None, hist_weights=False):
        hyperparams = self.variational.hyperparams.state_vardict()
        for k, v in hyperparams.items():
            hyperparams[k] = v.expand(1, *v.shape)

        guide = probtorch.Trace()
        if block is not None:
            subject = self.generative.block_subjects[block]
            task = self.generative.block_tasks[block]
            times = (0, self.num_times[block])
            blocks = [block]
            block_subjects = [self.generative.block_subjects[block]]
            block_tasks = [self.generative.block_tasks[block]]
            block_interactions = [self.generative.block_interactions[block]]
        else:
            times = (0, max(self.num_times))
            blocks = []
            block_subjects = self.generative.block_subjects
            block_tasks = self.generative.block_tasks
            block_interactions = self.generative.block_interactions

        for b in blocks:
            if subject is not None:
                guide.variable(
                    torch.distributions.Normal,
                    hyperparams['subject']['mu'][:, subject],
                    torch.exp(hyperparams['subject']['sigma'][:, subject]),
                    value=hyperparams['subject']['mu'][:, subject],
                    name='z^P_{%d,%d}' % (subject, b),
                )
                factor_centers_params = hyperparams['factor_centers']
                guide.variable(
                    torch.distributions.Normal,
                    factor_centers_params['mu'][:, subject],
                    torch.exp(factor_centers_params['sigma'][:, subject]),
                    value=factor_centers_params['mu'][:, subject],
                    name='FactorCenters%d' % b,
                )
                factor_log_widths_params = hyperparams['factor_log_widths']
                guide.variable(
                    torch.distributions.Normal,
                    factor_log_widths_params['mu'][:, subject],
                    torch.exp(factor_log_widths_params['sigma'][:, subject]),
                    value=factor_log_widths_params['mu'][:, subject],
                    name='FactorLogWidths%d' % b,
                )
            if interaction is not None:
                guide.variable(
                    torch.distributions.Normal,
                    hyperparams['interactions']['mu'][:, task],
                    torch.exp(hyperparams['interactions']['sigma'][:, task]),
                    value=hyperparams['interactions']['mu'][:, task],
                    name='z^I_{%d,%d}' % (interaction , b),
                )
            if self._time_series:
                for k, v in hyperparams['weights'].items():
                    hyperparams['weights'][k] = v[:, :, times[0]:times[1]]
                weights_params = hyperparams['weights']
                guide.variable(
                    torch.distributions.Normal,
                    weights_params['mu'][:, b],
                    torch.exp(weights_params['sigma'][:, b]),
                    value=weights_params['mu'][:, b],
                    name='Weights%d_%d-%d' % (b, times[0], times[1])
                )

        weights, factor_centers, factor_log_widths =\
            self.decoder(probtorch.Trace(), blocks, block_subjects, block_tasks, block_interactions,
                         hyperparams, times, guide=guide, num_particles=1)

        if block is not None:
            weights = weights[0]
            factor_centers = factor_centers[0]
            factor_log_widths = factor_log_widths[0]
        weights = weights.squeeze(0)
        factor_centers = factor_centers.squeeze(0)
        factor_log_widths = factor_log_widths.squeeze(0)

        if hist_weights:
            plt.hist(weights.view(weights.numel()).data.numpy())
            plt.show()

        result = {
            'weights': weights[times[0]:times[1]].data,
            'factors': tfa_models.radial_basis(self.voxel_locations,
                                               factor_centers.data,
                                               factor_log_widths.data),
            'factor_centers': factor_centers.data,
            'factor_log_widths': factor_log_widths.data,
        }
        if subject is not None:
            result['z^P_%d' % subject] = hyperparams['subject']['mu'][:, subject]
        if interaction is not None:
            result['z^I_{%d}' % interaction] = hyperparams['interactions']['mu'][:, interaction]
        return result

    def reconstruction(self, block=None, subject=None, task=None, t=0):
        results = self.results(block, subject, task)
        reconstruction = results['weights'] @ results['factors']

        image = utils.cmu2nii(reconstruction.numpy(),
                              self.voxel_locations.numpy(),
                              self._templates[block])
        if t is None:
            image_slice = nilearn.image.mean_img(image)
            reconstruction = reconstruction.mean(dim=0)
        else:
            image_slice = nilearn.image.index_img(image, t)
            reconstruction = reconstruction[t]
        return image_slice, reconstruction

    def reconstruction_diff(self, block, t=0, zscore_bound=3):
        results = self.results(block)
        reconstruction = results['weights'] @ results['factors']
        squared_diff = (self.voxel_activations[block] - reconstruction) ** 2

        if zscore_bound is None:
            zscore_bound = squared_diff.max().item()

        image = utils.cmu2nii(squared_diff.numpy(),
                              self.voxel_locations.numpy(),
                              self._templates[block])
        if t is None:
            image_slice = nilearn.image.mean_img(image)
            squared_diff = self.voxel_activations[block].mean(dim=0) -\
                           reconstruction.mean(dim=0)
        else:
            image_slice = nilearn.image.index_img(image, t)
            squared_diff = self.voxel_activations[block][t] - reconstruction[t]
        squared_diff = squared_diff ** 2

        return image_slice, squared_diff

    def plot_reconstruction_diff(self, block, filename='', show=True, t=0,
                                 plot_abs=False, labeler=lambda b: None,
                                 zscore_bound=3, **kwargs):
        if filename == '' and t is None:
            filename = '%s-%s_ntfa_reconstruction_diff.pdf'
            filename = filename % (self.common_name(), str(block))
        elif filename == '':
            filename = '%s-%s_ntfa_reconstruction_diff_tr%d.pdf'
            filename = filename % (self.common_name(), str(block), t)

        image_slice, diff = self.reconstruction_diff(block, t=t,
                                                     zscore_bound=zscore_bound)
        plot = niplot.plot_glass_brain(
            image_slice, plot_abs=plot_abs, colorbar=True, symmetric_cbar=False,
            title=utils.title_brain_plot(block, self._blocks[block], labeler, t,
                                         'Squared Residual'),
            vmin=0, vmax=zscore_bound ** 2, **kwargs,
        )

        logging.info(
            'Reconstruction Error (Frobenius Norm): %.8e out of %.8e',
            np.linalg.norm(diff.sqrt().numpy()),
            np.linalg.norm(self.voxel_activations[block].numpy())
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def normalize_activations(self):
        subject_runs = OrderedSet([(block.subject, block.run)
                                   for block in self._blocks])
        run_activations = {sr: None for sr in subject_runs}

        for block in range(len(self._blocks)):
            sr = (self._blocks[block].subject, self._blocks[block].run)
            if run_activations[sr] is None:
                run_activations[sr] = self.voxel_activations[block]
            else:
                run_activations[sr] = torch.cat((run_activations[sr],
                                                 self.voxel_activations[block]),
                                                dim=0)

        for sr in run_activations:
            run_activations[sr] = run_activations[sr].flatten()

        self.activation_normalizers =\
            [torch.abs(run_activations[(block.subject, block.run)]).max()
             for block in self._blocks]
        self.activation_sufficient_stats = [
            (torch.mean(run_activations[(block.subject, block.run)], dim=0),
             torch.std(run_activations[(block.subject, block.run)], dim=0))
            for block in self._blocks]
        return self.activation_normalizers

    def plot_factor_centers(self, block, filename='', show=True, t=None,
                            labeler=None, serialize_data=True):
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
            title=utils.title_brain_plot(block, self._blocks[block], labeler,
                                         None, 'Factor Centers'),
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
        if self.activation_normalizers is None:
            self.normalize_activations()

        image = utils.cmu2nii(self.voxel_activations[block].numpy(),
                              self.voxel_locations.numpy(),
                              self._templates[block])
        if t is None:
            image_slice = nilearn.image.mean_img(image)
        else:
            image_slice = nilearn.image.index_img(image, t)
        plot = niplot.plot_glass_brain(
            image_slice, plot_abs=plot_abs, colorbar=True, symmetric_cbar=True,
            title=utils.title_brain_plot(block, self._blocks[block], labeler, t),
            vmin=-zscore_bound, vmax=zscore_bound, **kwargs,
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def average_reconstruction_error(self, weighted=True,
                                     blocks_filter=lambda block: True):
        if self.activation_normalizers is None:
            self.normalize_activations()
        blocks = [block for block in range(self.num_blocks)
                  if blocks_filter(self._blocks[block])]

        if weighted:
            return utils.average_weighted_reconstruction_error(
                blocks, self.num_times, self.num_voxels,
                self.voxel_activations, self.results
            )
        else:
            return utils.average_reconstruction_error(
                blocks, self.voxel_activations, self.results
            )

    def plot_reconstruction(self, block=None, filename='', show=True,
                            plot_abs=False, t=0, labeler=None, zscore_bound=3,
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
        if block is None:
            block = np.random.choice(self.num_blocks, 1)[0]
        if self.activation_normalizers is None:
            self.normalize_activations()

        image_slice, reconstruction = self.reconstruction(block=block, t=t)
        plot = niplot.plot_glass_brain(
            image_slice, plot_abs=plot_abs, colorbar=True, symmetric_cbar=True,
            title=utils.title_brain_plot(block, self._blocks[block], labeler, t,
                                         'NeuralTFA'),
            vmin=-zscore_bound, vmax=zscore_bound, **kwargs,
        )

        activations = self.voxel_activations[block]
        if t:
            activations = activations[t]
        else:
            activations = activations.mean(dim=0)

        logging.info(
            'Reconstruction Error (Frobenius Norm): %.8e out of %.8e',
            np.linalg.norm((activations - reconstruction).numpy()),
            np.linalg.norm(self.voxel_activations[block].numpy())
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
        template = [i for (i, b) in enumerate(self._blocks)
                    if b.subject == subject][0]
        reconstruction = results['weights'] @ results['factors']
        if zscore_bound is None:
            zscore_bound = self.activation_normalizers[template]

        image = utils.cmu2nii(reconstruction.numpy(),
                              self.voxel_locations.numpy(),
                              self._templates[template])
        image_slice = nilearn.image.index_img(image, 0)

        if serialize_data:
            tensors_filename = os.path.splitext(filename)[0] + '.dat'
            tensors = {
                'reconstruction': reconstruction,
                'voxel_locations': self.voxel_locations,
                'template': self._templates[template],
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
        template = [i for (i, b) in enumerate(self._blocks)
                    if b.task == task][0]
        reconstruction = results['weights'] @ results['factors']
        if zscore_bound is None:
            zscore_bound = self.activation_normalizers[template]

        image = utils.cmu2nii(reconstruction.numpy(),
                              self.voxel_locations.numpy(),
                              self._templates[template])
        image_slice = nilearn.image.index_img(image, 0)

        if serialize_data:
            tensors_filename = os.path.splitext(filename)[0] + '.dat'
            tensors = {
                'reconstruction': reconstruction,
                'voxel_locations': self.voxel_locations,
                'template': self._templates[template],
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
        z_p_sigma = torch.exp(hyperparams['subject']['sigma'].data)
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
        z_p_sigma = torch.exp(hyperparams['subject']['sigma'].data)
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
        z_p_sigma = torch.exp(hyperparams['subject_weight']['sigma'].data)
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
                'z_pw': {'mu': z_p_mu, 'sigma': z_p_sigma},
                'palette': palette,
                'subject_colors': subject_colors,
                'labels': labels,
            }
            torch.save(tensors, tensors_filename)

        utils.embedding_clusters_fig(z_p_mu, z_p_sigma, subject_colors, 'z^PW',
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
        z_s_sigma = torch.exp(hyperparams['task']['sigma'].data)
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
        if self._common_name:
            return self._common_name
        return os.path.commonprefix([os.path.basename(b.filename)
                                     for b in self._blocks])

    def save_state(self, path='.', tag=''):
        name = self.common_name() + tag
        variational_state = self.variational.state_dict()
        torch.save(variational_state,
                   path + '/' + name + '.dtfa_guide')
        torch.save(self.decoder.state_dict(),
                   path + '/' + name + '.dtfa_model')

    def save(self, path='.'):
        name = self.common_name()
        torch.save(self.variational.state_dict(),
                   path + '/' + name + '.dtfa_guide')
        torch.save(self.decoder.state_dict(),
                   path + '/' + name + '.dtfa_model')
        with open(path + '/' + name + '.dtfa', 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    def load_state(self, basename):
        model_state = torch.load(basename + '.dtfa_model')
        self.decoder.load_state_dict(model_state)

        guide_state = torch.load(basename + '.dtfa_guide')
        self.variational.load_state_dict(guide_state)

    @classmethod
    def load(cls, basename):
        with open(basename + '.dtfa', 'rb') as pickle_file:
            dtfa = pickle.load(pickle_file)
        dtfa.load_state(basename)

        return dtfa

    def decoding_accuracy(self, labeler=lambda x: x, window_size=60):
        """
        :return: accuracy: a dict containing decoding accuracies for each task [activity,isfc,mixed]
        """
        tasks = np.unique([labeler(b.task) for b in self._blocks])
        group = {task: [] for task in tasks}
        accuracy = {task: {'node': [], 'isfc': [], 'mixed': [], 'kl': []}
                    for task in tasks}

        for (b, block) in enumerate(self._blocks):
            factorization = self.results(b)
            group[(block.task)].append(factorization['weights'])

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
        keys = np.unique([labeler(b.task) for b in self._blocks])
        group = {key: [] for key in keys}
        accuracy = {key: [] for key in keys}
        for key in keys:
            print(key)
            for n in range(self.num_blocks):
                if key == self._blocks[n].task:
                    self._blocks[n].load()
                    group[key].append(self._blocks[n].activations[:times[n], :])
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
