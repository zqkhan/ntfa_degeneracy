"""Perform hierarchical topographic factor analysis on a given fMRI data file."""

__author__ = 'Eli Sennesh', 'Zulqarnain Khan'
__email__ = 'e.sennesh@northeastern.edu', 'khan.zu@husky.neu.edu'

import collections
import datetime
import logging
import os
import pickle
import time

try:
    if __name__ == '__main__':
        import matplotlib
        matplotlib.use('TkAgg')
finally:
    import matplotlib.pyplot as plt

import hypertools as hyp
import nibabel as nib
import nilearn.image
import nilearn.plotting as niplot
import numpy as np
from ordered_set import OrderedSet
import scipy.io as sio
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.distributions as dists
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter
import torch.utils.data

import probtorch

from . import htfa_models
from . import niidb
from . import tfa
from . import tfa_models
from . import utils

class HierarchicalTopographicFactorAnalysis:
    """Overall container for a run of TFA"""
    def __init__(self, query, mask, num_factors=tfa_models.NUM_FACTORS,
                 query_name=None):
        self.num_factors = num_factors
        self.mask = mask
        self._common_name = query_name
        self._blocks = list(query)
        for block in self._blocks:
            block.load()
        self.num_blocks = len(self._blocks)
        self.voxel_activations = [block.activations for block in self._blocks]
        self.voxel_locations = self._blocks[0].locations
        self.task_list = [block.task for block in self._blocks]
        self.task_list = np.unique(self.task_list)
        self._templates = [block.filename for block in self._blocks]

        self.activation_normalizers = None

        # Pull out relevant dimensions: the number of time instants and the
        # number of voxels in each timewise "slice"
        self.num_times = [acts.shape[0] for acts in self.voxel_activations]
        self.num_voxels = [acts.shape[1] for acts in self.voxel_activations]

        self.enc = htfa_models.HTFAGuide(query, self.num_factors)
        self.dec = htfa_models.HTFAModel(self.voxel_locations, self.num_blocks,
                                         self.num_times, self.num_factors,
                                         volume=True)

    def num_parameters(self):
        return sum([param.numel() for param in self.enc.parameters()])

    def train(self, num_steps=10, learning_rate=tfa.LEARNING_RATE,
              log_level=logging.WARNING, num_particles=tfa_models.NUM_PARTICLES,
              batch_size=64, use_cuda=True, blocks_batch_size=4,
              checkpoint_steps=None, train_globals=True,
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
            batch_size=batch_size
        )
        enc = self.enc
        dec = self.dec
        if tfa.CUDA and use_cuda:
            enc.cuda()
            dec.cuda()
        param_groups = [{
            'params': [phi for phi in self.enc.parameters()
                       if phi.shape[0] == len(self._blocks)],
            'lr': learning_rate,
        }]
        if train_globals:
            param_groups.append({
                'params': [phi for phi in self.enc.parameters()
                           if phi.shape[0] != len(self._blocks)],
                'lr': learning_rate,
            })
        optimizer = torch.optim.Adam(param_groups, lr=learning_rate,
                                     amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=1e-1, min_lr=5e-5
        )
        enc.train()
        dec.train()

        free_energies = list(range(num_steps))
        rv_occurrences = collections.defaultdict(int)
        measure_occurrences = True

        for epoch in range(num_steps):
            start = time.time()
            epoch_free_energies = list(range(len(activations_loader)))

            for (batch, data) in enumerate(activations_loader):
                epoch_free_energies[batch] = 0.0
                block_batches = utils.chunks(list(range(len(training_blocks))),
                                             n=blocks_batch_size)
                for block_batch in block_batches:
                    activations = [{'Y': data[:, b, :]} for b in block_batch]
                    block_batch = [training_blocks[b][0] for b in block_batch]
                    if tfa.CUDA and use_cuda:
                        for acts in activations:
                            acts['Y'] = acts['Y'].cuda()
                        for b in block_batch:
                            dec.likelihoods[b].voxel_locations =\
                                self.voxel_locations.cuda()
                    trs = (batch * batch_size, None)
                    trs = (trs[0], trs[0] + activations[0]['Y'].shape[0])

                    optimizer.zero_grad()
                    q = probtorch.Trace()
                    enc(q, times=trs, num_particles=num_particles,
                        blocks=block_batch)
                    p = probtorch.Trace()
                    dec(p, times=trs, guide=q, observations=activations,
                        blocks=block_batch)

                    def block_rv_weight(node, prior=True):
                        result = 1.0
                        if measure_occurrences:
                            rv_occurrences[node] += 1
                        result /= rv_occurrences[node]
                        return result
                    free_energy, _, _ = tfa.hierarchical_free_energy(
                        q, p,
                        rv_weight=block_rv_weight,
                        num_particles=num_particles
                    )

                    free_energy.backward()
                    optimizer.step()
                    epoch_free_energies[batch] += free_energy

                    if tfa.CUDA and use_cuda:
                        del activations
                        for b in block_batch:
                            locs = dec.likelihoods[b].voxel_locations
                            dec.likelihoods[b].voxel_locations =\
                                locs.cpu()
                            del locs
                        torch.cuda.empty_cache()
                if tfa.CUDA and use_cuda:
                    epoch_free_energies[batch] = epoch_free_energies[batch].cpu().data.numpy()
                else:
                    epoch_free_energies[batch] = epoch_free_energies[batch].data.numpy()

            free_energies[epoch] = np.array(epoch_free_energies).sum(0)
            free_energies[epoch] = free_energies[epoch].sum(0)
            scheduler.step(free_energies[epoch])

            measure_occurrences = False

            end = time.time()
            msg = tfa.EPOCH_MSG % (epoch + 1, (end - start) * 1000, free_energies[epoch])
            logging.info(msg)
            if checkpoint_steps is not None and epoch % checkpoint_steps == 0:
                now = datetime.datetime.now()
                checkpoint_name = now.strftime(tfa.CHECKPOINT_TAG)
                logging.info('Saving checkpoint...')
                self.save_state(path='.', tag=checkpoint_name)

        if tfa.CUDA and use_cuda:
            dec.cpu()
            enc.cpu()

        now = datetime.datetime.now()
        checkpoint_name = now.strftime(tfa.CHECKPOINT_TAG)
        logging.info('Saving checkpoint...')
        self.save_state(path='.', tag=checkpoint_name)

        return np.vstack([free_energies])

    def free_energy(self, batch_size=64, use_cuda=True, blocks_batch_size=4,
                    blocks_filter=lambda block: True, num_particles=1,
                    sample_size=10):
        training_blocks = [(b, block) for (b, block) in enumerate(self._blocks)
                           if blocks_filter(block)]
        activations_loader = torch.utils.data.DataLoader(
            utils.TFADataset([block.activations
                              for (_, block) in training_blocks]),
            batch_size=batch_size,
            pin_memory=True,
        )
        enc = self.enc
        dec = self.dec
        log_likelihoods = torch.zeros(sample_size, len(activations_loader))
        prior_kls = torch.zeros(sample_size, len(activations_loader))
        if tfa.CUDA and use_cuda:
            enc.cuda()
            dec.cuda()
            cuda_locations = self.voxel_locations.cuda()
            log_likelihoods = log_likelihoods.to(cuda_locations)
            prior_kls = prior_kls.to(cuda_locations)

        for k in range(sample_size // num_particles):
            for (batch, data) in enumerate(activations_loader):
                block_batches = utils.chunks(list(range(len(training_blocks))),
                                             n=blocks_batch_size)
                for block_batch in block_batches:
                    activations = [{'Y': data[:, b, :]} for b in block_batch]
                    block_batch = [training_blocks[b][0] for b in block_batch]
                    if tfa.CUDA and use_cuda:
                        for b in block_batch:
                            dec.likelihoods[b].voxel_locations = cuda_locations
                        for acts in activations:
                            acts['Y'] = acts['Y'].cuda()
                    trs = (batch * batch_size, None)
                    trs = (trs[0], trs[0] + activations[0]['Y'].shape[0])

                    q = probtorch.Trace()
                    enc(q, times=trs, num_particles=num_particles,
                        blocks=block_batch)
                    p = probtorch.Trace()
                    dec(p, times=trs, guide=q, observations=activations,
                        blocks=block_batch)

                    _, ll, prior_kl = tfa.hierarchical_free_energy(
                        q, p, num_particles=num_particles
                    )

                    start = k * num_particles
                    end = (k + 1) * num_particles
                    log_likelihoods[start:end, batch] += ll.detach()
                    prior_kls[start:end, batch] += prior_kl.detach()

                    if tfa.CUDA and use_cuda:
                        del activations
                        for b in block_batch:
                            dec.likelihoods[b].voxel_locations =\
                                self.voxel_locations
                        torch.cuda.empty_cache()

        if tfa.CUDA and use_cuda:
            dec.cpu()
            enc.cpu()
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

    def save(self, out_dir='.'):
        '''Save a HierarchicalTopographicFactorAnalysis'''
        with open(out_dir + '/' + self._name + '.htfa', 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        '''Load a saved HierarchicalTopographicFactorAnalysis from a file'''
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def results(self, block=None):
        """Return the inferred posterior parameters and reconstruction
           components"""
        hyperparams = self.enc.hyperparams.state_vardict()

        if block is not None:
            centers = hyperparams['block']['factor_centers']['mu'][block].data
            log_widths = hyperparams['block']['factor_log_widths']['mu'][block].data
            weights = hyperparams['block']['weights']['mu'][block]\
                                 [:self.num_times[block]].data

            result = {
                'factors': tfa_models.radial_basis(self.voxel_locations,
                                                   centers, log_widths),
                'factor_centers': centers,
                'factor_log_widths': log_widths,
                'weights': weights,
            }
        else:
            centers = hyperparams['template']['factor_centers']['mu'].data
            log_widths = hyperparams['template']['factor_log_widths']['mu'].data
            weights = hyperparams['block']['weights']['mu'].mean(dim=0)

            result = {
                'factors': tfa_models.radial_basis(self.voxel_locations,
                                                   centers, log_widths),
                'factor_centers': centers,
                'factor_log_widths': log_widths,
                'weights': weights[:max(self.num_times)],
            }

        return result

    def visualize_factor_template(self, filename=None, show=True,
                                  hist_log_widths=True, **kwargs):
        results = self.results(block=None)
        centers = results['factor_centers']
        log_widths = results['factor_log_widths']
        widths = torch.exp(log_widths)

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

    def normalize_activations(self):
        subject_runs = OrderedSet([(block.subject, block.run)
                                   for block in self._blocks])
        subject_run_normalizers = {sr: 0 for sr in subject_runs}

        for block in range(len(self._blocks)):
            sr = (self._blocks[block].subject, self._blocks[block].run)
            subject_run_normalizers[sr] = max(
                subject_run_normalizers[sr],
                torch.abs(self.voxel_activations[block]).max()
            )

        self.activation_normalizers =\
            [subject_run_normalizers[(block.subject, block.run)]
             for block in self._blocks]
        return self.activation_normalizers

    def plot_voxels(self, block=None):
        if block:
            hyp.plot(self.voxel_locations.numpy(), 'k.')
        else:
            for b in range(self.num_blocks):
                hyp.plot(self.voxel_locations.numpy(), 'k.')

    def plot_factor_centers(self, block=None, filename=None, show=True, t=None,
                            labeler=None):
        if labeler is None:
            labeler = lambda b: b.task
        hyperparams = self.enc.hyperparams.state_vardict()

        if block is not None:
            factor_centers =\
                hyperparams['block']['factor_centers']['mu'][block]
            factor_log_widths =\
                hyperparams['block']['factor_log_widths']['mu'][block]
        else:
            factor_centers =\
                hyperparams['template']['factor_centers']['mu']
            factor_log_widths =\
                hyperparams['template']['factor_log_widths']['mu']

        if block is not None:
            title = "Block %d (Participant %s, Run %d, Stimulus: %s)" %\
                  (block, self._blocks[block].subject, self._blocks[block].run,
                   labeler(self._blocks[block]))
        else:
            title = 'Average block'

        centers_sizes = np.repeat([50], self.num_factors)
        sizes = torch.exp(factor_log_widths.data).numpy()

        centers = factor_centers.data.numpy()

        plot = niplot.plot_connectome(
            np.eye(self.num_factors * 2),
            np.vstack([centers, centers]),
            node_size=np.vstack([sizes, centers_sizes]),
            title=title
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def plot_original_brain(self, block=None, filename='', show=True,
                            plot_abs=False, t=0, labeler=None, zscore_bound=3,
                            **kwargs):
        if block is None:
            block = np.random.choice(self.num_blocks, 1)[0]
        if self.activation_normalizers is None:
            self.normalize_activations()
        if zscore_bound is None:
            zscore_bound = self.activation_normalizers[block]
        if labeler is None:
            labeler = lambda b: None
        if filename == '' and t is None:
            filename = '%s-%s_original_brain.pdf' % (self.common_name(),
                                                     str(block))
        elif filename == '':
            filename = '%s-%s_original_brain_tr%d.pdf'
            filename = filename % (self.common_name(), str(block), t)

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

    def posterior_reconstruction(self, block=None, t=0):
        results = self.results(block)

        factors = tfa_models.radial_basis(
            self.voxel_locations, results['factor_centers'],
            results['factor_log_widths']
        )
        if block:
            times = (0, self.num_times[block])
            template = self._templates[block]
        else:
            times = (0, max(self.num_times))
            template = self._templates[0]
        reconstruction = results['weights'][times[0]:times[1], :] @ factors
        reconstruction = reconstruction.detach()

        image = utils.cmu2nii(reconstruction.numpy(),
                              self.voxel_locations.numpy(), template)
        if t is None:
            image_slice = nilearn.image.mean_img(image)
            reconstruction = reconstruction.mean(dim=0)
        else:
            image_slice = nilearn.image.index_img(image, t)
            reconstruction = reconstruction[t]

        return image_slice, reconstruction

    def posterior_predictive_reconstruction(self):
        return self.posterior_reconstruction(None, None)

    def plot_reconstruction(self, block=None, filename='', show=True,
                            plot_abs=False, t=0, labeler=None, zscore_bound=3,
                            blocks_filter=lambda block: True, **kwargs):
        if self.activation_normalizers is None:
            self.normalize_activations()
        if zscore_bound is None and block:
            zscore_bound = self.activation_normalizers[block]
        if labeler is None:
            labeler = lambda b: None
        if filename == '' and t is None:
            filename = '%s-%s_htfa_reconstruction.pdf' % (self.common_name(),
                                                          str(block))
        elif filename == '':
            filename = '%s-%s_htfa_reconstruction_tr%d.pdf'
            filename = filename % (self.common_name(), str(block), t)

        if blocks_filter(self._blocks[block]):
            image_slice, reconstruction = self.posterior_reconstruction(
                block=block, t=t
            )
        else:
            image_slice, reconstruction =\
                self.posterior_predictive_reconstruction()
        plot = niplot.plot_glass_brain(
            image_slice, plot_abs=plot_abs, colorbar=True, symmetric_cbar=True,
            title=utils.title_brain_plot(block, self._blocks[block], labeler, t,
                                         'HTFA'),
            vmin=-zscore_bound, vmax=zscore_bound, **kwargs,
        )

        logging.info(
            'Reconstruction Error (Frobenius Norm): %.8e out of %.8e',
            np.linalg.norm(
                (reconstruction - self.voxel_activations[block]).numpy()
            ),
            np.linalg.norm(self.voxel_activations[block].numpy())
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def plot_reconstruction_diff(self, block, filename='', show=True,
                                 plot_abs=False, t=0, labeler=lambda b: None,
                                 zscore_bound=3, **kwargs):
        if filename == '' and t is None:
            filename = '%s-%s_htfa_reconstruction_diff.pdf'
            filename = filename % (self.common_name(), str(block))
        elif filename == '':
            filename = '%s-%s_htfa_reconstruction_diff_tr%d.pdf'
            filename = filename % (self.common_name(), str(block), t)

        results = self.results(block)
        factor_centers = results['factor_centers']
        factor_log_widths = results['factor_log_widths']
        if block is not None:
            weights = results['weights']
        else:
            block = np.random.choice(self.num_blocks, 1)[0]
            weights = self.enc.hyperparams.state_vardict()['block']['weights']['mu'][block]

        factors = tfa_models.radial_basis(
            self.voxel_locations, factor_centers,
            factor_log_widths
        )
        times = (0, self.voxel_activations[block].shape[0])
        reconstruction = weights[times[0]:times[1], :] @ factors

        diff = self.voxel_activations[block] - reconstruction
        if zscore_bound is None:
            zscore_bound = diff.max().item()
        image = utils.cmu2nii(diff.numpy() ** 2, self.voxel_locations.numpy(),
                              self._templates[block])

        if t is None:
            image_slice = nilearn.image.mean_img(image)
        else:
            image_slice = nilearn.image.index_img(image, t)
        plot = niplot.plot_glass_brain(
            image_slice, plot_abs=plot_abs, colorbar=True, symmetric_cbar=False,
            title=utils.title_brain_plot(block, self._blocks[block], labeler, t,
                                         'Squared Residual'),
            vmin=0, vmax=zscore_bound ** 2, **kwargs,
        )

        logging.info(
            'Reconstruction Error (Frobenius Norm): %.8e out of %.8e',
            np.linalg.norm(diff.numpy()),
            np.linalg.norm(self.voxel_activations[block].numpy())
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def common_name(self):
        if self._common_name:
            return self._common_name
        return os.path.commonprefix([os.path.basename(b.filename)
                                     for b in self._blocks])

    def save_state(self, path='.', tag=''):
        name = self.common_name() + tag
        variational_state = self.enc.state_dict()
        torch.save(variational_state,
                   path + '/' + name + '.htfa_guide')
        torch.save(self.dec.state_dict(),
                   path + '/' + name + '.htfa_model')

    def load_state(self, basename):
        model_state = torch.load(basename + '.htfa_model')
        self.dec.load_state_dict(model_state)

        guide_state = torch.load(basename + '.htfa_guide')
        self.enc.load_state_dict(guide_state)

    def scatter_factor_embedding(self, labeler=None, filename=None, show=True,
                                 xlims=None, ylims=None, figsize=None,
                                 embedding=TSNE):
        factor_centers_map = self.enc.hyperparams.block__factor_centers__mu.data.numpy()
        factor_widths_map = self.enc.hyperparams.block__factor_log_widths__mu.data.numpy()
        factors_map = np.concatenate(
            (np.expand_dims(factor_widths_map, 2), factor_centers_map),
            axis=2
        )
        factors_map = np.reshape(factors_map, newshape=(self.num_blocks, self.num_factors * 4))
        X = StandardScaler().fit_transform(factors_map)
        if embedding == 'TSNE':
            z_f = TSNE(n_components=2).fit_transform(X)
        else:
            z_f = PCA(n_components=2).fit_transform(X)


        if labeler is None:
            labeler = lambda b: b.default_label()
        labels = [labeler(b) for b in self._blocks]
        all_labels = [l for l in labels if l is not None]
        all_labels = np.unique(all_labels)
        palette = dict(zip(all_labels, utils.compose_palette(len(all_labels))))

        z_fs = [z_f[b] for b in range(self.num_blocks) if labels[b] is not None]
        z_fs = np.stack(z_fs)
        block_colors = [palette[labels[b]] for b in range(self.num_blocks)
                        if labels[b] is not None]

        fig = plt.figure(1, figsize=figsize)
        ax = fig.add_subplot(111, facecolor='white')
        fig.axes[0].set_xlabel('$z^F_1$')
        if xlims is not None:
            fig.axes[0].set_xlim(*xlims)
        fig.axes[0].set_ylabel('$z^F_2$')
        if ylims is not None:
            fig.axes[0].set_ylim(*ylims)
        fig.axes[0].set_title('Factor Embeddings')
        ax.scatter(x=z_fs[:, 0], y=z_fs[:, 1], c=block_colors)
        utils.palette_legend(list(palette.keys()), list(palette.values()))

        if filename is not None:
            fig.savefig(filename)
        if show:
            plt.show()

    def scatter_weight_embedding(self, labeler=None, filename=None, show=True,
                                 xlims=None, ylims=None, figsize=None,
                                 embedding='TSNE'):
        weight_map = self.enc.hyperparams.block__weights__mu.data.numpy()
        weight_map = np.reshape(weight_map,
                                newshape=(self.num_blocks,
                                          self.num_factors * weight_map.shape[1]))
        X = StandardScaler().fit_transform(weight_map)
        if embedding == 'TSNE':
            z_w = TSNE(n_components=2).fit_transform(X)
        else:
            z_w = PCA(n_components=2).fit_transform(X)

        if labeler is None:
            labeler = lambda b: b.default_label()
        labels = [labeler(b) for b in self._blocks]
        all_labels = [l for l in labels if l is not None]
        all_labels = np.unique(all_labels)
        palette = dict(zip(all_labels, utils.compose_palette(len(all_labels))))

        z_ws = [z_w[b] for b in range(self.num_blocks) if labels[b] is not None]
        z_ws = np.stack(z_ws)
        block_colors = [palette[labels[b]] for b in range(self.num_blocks)
                        if labels[b] is not None]

        fig = plt.figure(1, figsize=figsize)
        ax = fig.add_subplot(111, facecolor='white')
        fig.axes[0].set_xlabel('$z^W_1$')
        if xlims is not None:
            fig.axes[0].set_xlim(*xlims)
        fig.axes[0].set_ylabel('$z^W_2$')
        if ylims is not None:
            fig.axes[0].set_ylim(*ylims)
        fig.axes[0].set_title('Weight Embeddings')
        ax.scatter(x=z_ws[:, 0], y=z_ws[:, 1], c=block_colors)
        utils.palette_legend(list(palette.keys()), list(palette.values()))

        if filename is not None:
            fig.savefig(filename)
        if show:
            plt.show()

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

    def decoding_accuracy(self, restvtask=False, window_size=5):
        """
        :return: accuracy: a dict containing decoding accuracies for each task [activity,isfc,mixed]
        """
        W = self.enc.hyperparams.block__weights__mu.data

        if restvtask:
            keys = ['rest', 'task']
            group = {key: [] for key in keys}
            # accuracy = {key:[] for key in keys}
            accuracy = {task: {'node': [], 'isfc': [], 'mixed': [], 'kl': []}
                        for task in keys}
            for key in keys:
                print(key)
                for n in range(self.num_blocks):
                    if key in self._blocks[n].task:
                        group[key].append(W[n, :, :])
                    else:
                        group['task'].append(W[n, :, :])
                group[key] = np.rollaxis(np.dstack(group[key]), -1)
                if group[key].shape[0] < 2:
                    raise ValueError('not enough subjects for the task: ' + key)
                else:
                    G1 = group[key][:int(group[key].shape[0] / 2), :, :]
                    G2 = group[key][int(group[key].shape[0] / 2):, :, :]
                    node_accuracy, node_correlation = utils.get_decoding_accuracy(G1, G2, window_size)
                    accuracy[key]['node'].append(node_accuracy)
                    isfc_accuracy, isfc_correlation = utils.get_isfc_decoding_accuracy(G1, G2, window_size)
                    accuracy[key]['isfc'].append(isfc_accuracy)
                    accuracy[key]['mixed'].append(utils.get_mixed_decoding_accuracy(node_correlation,isfc_correlation))
                    accuracy[key]['kl'].append(utils.get_kl_decoding_accuracy(G1, G2, window_size))
        else:
            keys = self.task_list
            group = {key: [] for key in keys}
            # accuracy = {key: [] for key in keys}
            accuracy = {task: {'node': [], 'isfc': [], 'mixed': [], 'kl': []}
                        for task in keys}
            for key in keys:
                print(key)
                for n in range(self.num_blocks):
                    if key == self._blocks[n].task:
                        group[key].append(W[n, :, :])
                group[key] = np.rollaxis(np.dstack(group[key]), -1)
                if group[key].shape[0] < 2:
                    raise ValueError('not enough subjects for the task: ' + key)

                else:
                    G1 = group[key][:int(group[key].shape[0] / 2), :, :]
                    G2 = group[key][int(group[key].shape[0] / 2):, :, :]
                    node_accuracy, node_correlation = utils.get_decoding_accuracy(G1, G2, window_size)
                    accuracy[key]['node'].append(node_accuracy)
                    isfc_accuracy, isfc_correlation = utils.get_isfc_decoding_accuracy(G1, G2, window_size)
                    accuracy[key]['isfc'].append(isfc_accuracy)
                    accuracy[key]['mixed'].append(utils.get_mixed_decoding_accuracy(node_correlation,isfc_correlation))
                    accuracy[key]['kl'].append(utils.get_kl_decoding_accuracy(G1, G2, window_size))
        return accuracy

    def voxel_decoding_accuracy(self,restvtask=False,window_size=5):
        times = self.num_times
        if restvtask:
            keys = ['rest', 'task']
            group = {key: [] for key in keys}
            accuracy = {key:[] for key in keys}
            for key in keys:
                print (key)
                for n in range(self.num_blocks):
                    if key in self._blocks[n].task:
                        group[key].append(self._blocks[n].activations[:times[n],:])
                    else:
                        group['task'].append(self._blocks[n].activations[:times[n],:])
                group[key] = np.rollaxis(np.dstack(group[key]), -1)
                if len(group[key]) < 2:
                    raise ValueError('not enough subjects for the task: ' + key)
                else:
                    G1 = group[key][:int(group[key].shape[0] / 2), :, :]
                    G2 = group[key][int(group[key].shape[0] / 2):, :, :]
                    accuracy[key].append(utils.get_decoding_accuracy(G1, G2, window_size))
        else:
            keys = self.task_list
            group = {key: [] for key in keys}
            accuracy = {key: [] for key in keys}
            for key in keys:
                print (key)
                for n in range(self.num_blocks):
                    if key == self._blocks[n].task:
                        group[key].append(self._blocks[n].activations[:times[n],:])
                group[key] = np.rollaxis(np.dstack(group[key]), -1)
                if group[key].shape[0] < 2:
                    raise ValueError('not enough subjects for the task: ' + key)

                else:
                    G1 = group[key][:int(group[key].shape[0] / 2), :, :]
                    G2 = group[key][int(group[key].shape[0] / 2):, :, :]
                    accuracy[key].append(utils.get_decoding_accuracy(G1, G2, window_size)[0])
        return accuracy
