"""Deep factor analysis models as ProbTorch modules"""

__author__ = ('Jan-Willem van de Meent',
              'Eli Sennesh',
              'Zulqarnain Khan')
__email__ = ('j.vandemeent@northeastern.edu',
             'e.sennesh@northeastern.edu',
             'khan.zu@husky.neu.edu')

import collections

import numpy as np
import scipy
import torch
import torch.distributions as dists
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.functional import softplus
import torch.utils.data

import probtorch

from . import htfa_models
from . import tfa_models
from . import utils

class DeepTFAGenerativeHyperparams(tfa_models.HyperParams):
    def __init__(self, num_subjects, num_tasks, num_interactions, embedding_dim=2, voxel_noise=tfa_models.VOXEL_NOISE):
        self.num_subjects = num_subjects
        self.num_tasks = num_tasks
        self.num_interactions = num_subjects * num_tasks
        self.embedding_dim = embedding_dim

        params = utils.vardict({
            'subject': {
                'mu': torch.zeros(self.num_subjects, self.embedding_dim),
                'log_sigma': torch.ones(self.num_subjects, self.embedding_dim).log(),
            },
            'subject_weight': {
                'mu': torch.zeros(self.num_subjects, self.embedding_dim),
                'log_sigma': torch.ones(self.num_subjects, self.embedding_dim).log(),
            },
            'task': {
                'mu': torch.zeros(self.num_tasks, self.embedding_dim),
                'log_sigma': torch.ones(self.num_tasks, self.embedding_dim).log(),
            },
            'interaction': {
                'mu': torch.zeros(self.num_subjects * self.num_tasks, self.embedding_dim),
                'log_sigma': torch.ones(self.num_subjects * self.num_tasks, self.embedding_dim).log(),
            },
            'voxel_noise': (torch.ones(1) * voxel_noise).log(), ##denominated in log_sigma
        })

        super(self.__class__, self).__init__(params, guide=False)

class DeepTFAGuideHyperparams(tfa_models.HyperParams):
    def __init__(self, num_blocks, num_times, num_factors, num_subjects,
                 num_tasks, num_interactions, hyper_means, embedding_dim=2, time_series=True):
        self.num_blocks = num_blocks
        self.num_subjects = num_subjects
        self.num_interactions = num_subjects * num_tasks
        self.num_tasks = num_tasks
        self.num_times = max(num_times)
        self._num_factors = num_factors
        self.embedding_dim = embedding_dim

        params = utils.vardict({
            'subject': {
                'mu': torch.zeros(self.num_subjects, self.embedding_dim),
                'log_sigma': torch.ones(self.num_subjects, self.embedding_dim).log(),
            },
            'subject_weight': {
                'mu': torch.zeros(self.num_subjects, self.embedding_dim),
                'log_sigma': torch.ones(self.num_subjects, self.embedding_dim).log(),
            },
            'task': {
                'mu': torch.zeros(self.num_tasks, self.embedding_dim),
                'log_sigma': torch.ones(self.num_tasks, self.embedding_dim).log(),
            },
            'interaction': {
                'mu': torch.zeros(self.num_subjects * self.num_tasks, self.embedding_dim),
                'log_sigma': torch.ones(self.num_subjects * self.num_tasks, self.embedding_dim).log(),
            },
            'factor_centers': {
                'mu': hyper_means['factor_centers'].expand(self.num_subjects,
                                                           self._num_factors,
                                                           3),
                'log_sigma': torch.zeros(self.num_subjects, self._num_factors,
                                         3),
            },
            'factor_log_widths': {
                'mu': hyper_means['factor_log_widths'].expand(
                    self.num_subjects, self._num_factors
                ),
                'log_sigma': torch.zeros(self.num_subjects, self._num_factors) +\
                             hyper_means['factor_log_widths'].std().log(),
            },
        })
        if time_series:
            params['weights'] = {
                'mu': torch.zeros(self.num_blocks, self.num_times,
                                  self._num_factors),
                'log_sigma': torch.zeros(self.num_blocks, self.num_times,
                                         self._num_factors),
            }

        super(self.__class__, self).__init__(params, guide=True)

class DeepTFADecoder(nn.Module):
    """Neural network module mapping from embeddings to a topographic factor
       analysis"""
    def __init__(self, num_factors, locations, 
                 embedding_dim=2, time_series=True, volume=None,
                 linear=''):
        # linear = string characters to indicate which embeddings to weights
        # should be a linear mapping
        
        super(DeepTFADecoder, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_factors = num_factors
        self._time_series = time_series

        center, center_sigma = utils.brain_centroid(locations)
        center_sigma = center_sigma.sum(dim=1)
        hull = scipy.spatial.ConvexHull(locations)
        coefficient = 1.0
        if volume is not None:
            coefficient = np.cbrt(hull.volume / self._num_factors)

        self.factors_embedding = nn.Sequential(
            nn.Linear(self._embedding_dim, self._embedding_dim * 2),
            nn.PReLU(),
            nn.Linear(self._embedding_dim * 2, self._embedding_dim * 4),
            nn.PReLU(),
            nn.Linear(self._embedding_dim * 4, self._num_factors * 4 * 2),
        )
        factor_loc = torch.cat(
            (center.expand(self._num_factors, 3),
             torch.ones(self._num_factors, 1) * np.log(coefficient)),
            dim=-1
        )
        factor_log_scale = torch.cat(
            (torch.log(center_sigma / coefficient).expand(
                self._num_factors, 3
            ), torch.zeros(self._num_factors, 1)),
            dim=-1
        )
        self.factors_embedding[-1].bias = nn.Parameter(
            torch.stack((factor_loc, factor_log_scale), dim=-1).reshape(
                self._num_factors * 4 * 2
            )
        )
        self.factors_skip = nn.Linear(self._embedding_dim, self._num_factors * 4 * 2)
        if locations is not None:
            self.register_buffer('locations_min',
                                 torch.min(locations, dim=0)[0])
            self.register_buffer('locations_max',
                                 torch.max(locations, dim=0)[0])

        if 'P' in linear:
            self.participant_weights_embedding = nn.Sequential(
                nn.Linear(self._embedding_dim, self._num_factors * 2)
            )
        else:
            self.participant_weights_embedding = nn.Sequential(
                nn.Linear(self._embedding_dim, self._embedding_dim * 4),
                nn.PReLU(),
                nn.Linear(self._embedding_dim * 4, self._num_factors * 2)
            )
        if 'S' in linear:
            self.stimulus_weights_embedding = nn.Sequential(
                nn.Linear(self._embedding_dim, self._num_factors * 2)
            )
        else:
            self.stimulus_weights_embedding = nn.Sequential(
                nn.Linear(self._embedding_dim, self._embedding_dim * 4),
                nn.PReLU(),
                nn.Linear(self._embedding_dim * 4, self._num_factors * 2)
            )
        if 'C' in linear:
            self.weights_embedding = nn.Sequential(
                nn.Linear(self._embedding_dim, self._num_factors * 2)
            )
        else:      
            self.weights_embedding = nn.Sequential(
                nn.Linear(self._embedding_dim, self._embedding_dim * 4),
                nn.PReLU(),
                nn.Linear(self._embedding_dim * 4, self._embedding_dim * 8),
                nn.PReLU(),
                nn.Linear(self._embedding_dim * 8, self._num_factors * 2),
            )
        
    def _predict_param(self, params, param, index, predictions, name, trace,
                       predict=True, guide=None, use_mean=False):
        if name in trace:
            return trace[name].value
        if predict:
            mu = predictions.select(-1, 0)
            log_sigma = predictions.select(-1, 1)
        else:
            mu = params[param]['mu']
            log_sigma = params[param]['log_sigma']
            if index is None:
                mu = mu.mean(dim=1)
                log_sigma = log_sigma.mean(dim=1)
            if isinstance(index, tuple):
                mu = mu[:, index[0], index[1]]
                log_sigma = log_sigma[:, index[0], index[1]]
            else:
                mu = mu[:, index]
                log_sigma = log_sigma[:, index]

        if use_mean:
            result = trace.normal(mu, torch.exp(log_sigma),
                                  value=mu, name=name)
        else:
            result = trace.normal(mu, torch.exp(log_sigma),
                                  value=utils.clamped(name, guide), name=name)
        return result

    def forward(self, trace, blocks, subjects, tasks, interactions, params, times, guide=None,
                generative=False,
                ablate_subjects=False, ablate_tasks=False, custom_interaction=None,
                predictive=False, subjects_factors=None, use_mean=False):
        origin = torch.zeros(params['subject']['mu'].shape[0], len(blocks),
                             self._embedding_dim)
        origin = origin.to(params['subject']['mu'])
        if subjects_factors is None:
            subjects_factors = subjects
        if subjects is not None:
            subject_embed = self._predict_param(
                params, 'subject', subjects_factors, None,
                'z^PF', trace, False, guide, use_mean=(use_mean) and not (generative),
            )
            subject_weight_embed = self._predict_param(
                params, 'subject_weight', subjects, None,
                'z^PW', trace, False, guide, use_mean=(use_mean) and not (generative),
            )
        else:
            subject_embed = origin
            subject_weight_embed = origin
        if tasks is not None:
            task_embed = self._predict_param(params, 'task', tasks, None, 'z^S',
                                             trace, False, guide, use_mean=(use_mean) and not (generative),)
        else:
            task_embed = origin

        if interactions is not None:
            interaction_embed = self._predict_param(params, 'interaction', interactions, None, 'z^I',
                                                    trace, False, guide, use_mean=(use_mean) and not (generative),)
        else:
            interaction_embed = origin

        if ablate_subjects:
            subject_weight_embed = torch.zeros_like(task_embed)
        elif ablate_tasks:
            task_embed = torch.zeros_like(subject_weight_embed)
        factor_params = (self.factors_embedding(subject_embed)).view(
            -1, self._num_factors, 4, 2
        )
        centers_predictions = factor_params[:, :, :3]
        log_widths_predictions = factor_params[:, :, 3]

        ### defining W = W_p + W_s + W_ps, which means adding their means and variances. the sqrt/logs/exp are because
        ### everything is defined in terms of log_sigmas.
        participant_weight_predictions_mu = \
            self.participant_weights_embedding(subject_weight_embed).view(-1, self._num_factors, 2)[:, :, 0]
        stimulus_weight_predictions_mu = \
            self.stimulus_weights_embedding(task_embed).view(-1, self._num_factors, 2)[:, :, 0]
        interaction_weight_predictions_mu = \
            self.weights_embedding(interaction_embed).view(-1, self._num_factors, 2)[:, :, 0]

        participant_weight_predictions_sigma = \
            self.participant_weights_embedding(subject_weight_embed).view(-1, self._num_factors, 2)[:, :, 1]
        stimulus_weight_predictions_sigma = \
            self.stimulus_weights_embedding(task_embed).view(-1, self._num_factors, 2)[:, :, 1]
        interaction_weight_predictions_sigma = \
            self.weights_embedding(interaction_embed).view(-1, self._num_factors, 2)[:, :, 1]

        participant_weight_predictions_sigma = torch.exp(participant_weight_predictions_sigma)
        stimulus_weight_predictions_sigma = torch.exp(stimulus_weight_predictions_sigma)
        interaction_weight_predictions_sigma = torch.exp(interaction_weight_predictions_sigma)

        weight_predictions = (self.weights_embedding(interaction_embed)).view(
            interaction_embed.shape[0], interaction_embed.shape[1], self._num_factors, 2
        )
        weight_predictions[:, :, :, 0] = participant_weight_predictions_mu + \
                                         stimulus_weight_predictions_mu + \
                                         interaction_weight_predictions_mu

        weight_predictions[:, :, :, 1] = torch.log(torch.sqrt(participant_weight_predictions_sigma ** 2\
                                         + stimulus_weight_predictions_sigma ** 2\
                                         + interaction_weight_predictions_sigma ** 2))

        weight_predictions = weight_predictions.unsqueeze(2).expand(
            interaction_embed.shape[0], interaction_embed.shape[1], len(times),
            self._num_factors, 2
        )
        centers_predictions = self._predict_param(
            params, 'factor_centers', subjects, centers_predictions.unsqueeze(0),
            'FactorCenters', trace, predict=generative, guide=guide, use_mean=(use_mean) and not (generative),
        )
        if 'locations_min' in self._buffers:
            centers_predictions = utils.clamp_locations(centers_predictions,
                                                        self.locations_min,
                                                        self.locations_max)
        log_widths_predictions = self._predict_param(
            params, 'factor_log_widths', subjects, log_widths_predictions.unsqueeze(0),
            'FactorLogWidths', trace, predict=generative, guide=guide, use_mean=(use_mean) and not (generative),
        )

        if generative or predictive: # or ablate_tasks or ablate_subjects or (custom_interaction is not None):
            _, block_indices = blocks.unique(return_inverse=True)
            time_idx = torch.arange(len(times), dtype=torch.long)
            weight_predictions = weight_predictions[:, block_indices, time_idx]
        weight_predictions = self._predict_param(
            params, 'weights', (blocks, times), weight_predictions,
            'Weights_%s' % [t.item() for t in times], trace,
            predict=(generative) or (predictive) or (blocks < 0).any()
                    or not self._time_series,
            guide=guide,
        )

        return weight_predictions, centers_predictions, log_widths_predictions, \
               participant_weight_predictions_mu, stimulus_weight_predictions_mu, interaction_weight_predictions_mu

class DeepTFAGuide(nn.Module):
    """Variational guide for deep topographic factor analysis"""
    def __init__(self, num_factors, block_subjects, block_tasks, block_interactions, num_blocks=1,
                 num_times=[1], embedding_dim=2, hyper_means=None,
                 time_series=True):
        super(self.__class__, self).__init__()
        self._num_blocks = num_blocks
        self._num_times = num_times
        self._num_factors = num_factors
        self._embedding_dim = embedding_dim
        self._time_series = time_series

        self.register_buffer('block_subjects', torch.tensor(block_subjects,
                                                            dtype=torch.long),
                             persistent=False)
        self.register_buffer('block_tasks', torch.tensor(block_tasks,
                                                         dtype=torch.long),
                             persistent=False)
        self.register_buffer('block_interactions', torch.tensor(block_interactions,
                                                         dtype=torch.long),
                             persistent=False)
        num_subjects = len(self.block_subjects.unique())
        num_tasks = len(self.block_tasks.unique())
        num_interactions = len(self.block_interactions.unique())

        self.hyperparams = DeepTFAGuideHyperparams(self._num_blocks,
                                                   self._num_times,
                                                   self._num_factors,
                                                   num_subjects, num_tasks, num_interactions,
                                                   hyper_means,
                                                   embedding_dim, time_series)

    def forward(self, decoder, trace, times=None, blocks=None, params=None,
                num_particles=tfa_models.NUM_PARTICLES, ablate_subjects=False, ablate_tasks=False,
                custom_interaction=None, predictive=False, block_subjects_factors=None, use_mean=False):
        if params is None:
            params = self.hyperparams.state_vardict(num_particles)
        if blocks is None:
            blocks = torch.arange(self._num_blocks)

        unique_blocks = blocks.unique()
        block_subjects = self.block_subjects[unique_blocks]
        block_tasks = self.block_tasks[unique_blocks]
        block_interactions = self.block_interactions[unique_blocks]

        if block_subjects_factors is None:
            block_subjects_factors = block_subjects
        else:
            block_subjects_factors = self.block_subjects[block_subjects_factors.unique()]

        return decoder(trace, blocks, block_subjects, block_tasks, block_interactions, params,
                       times=times, ablate_subjects=ablate_subjects, ablate_tasks=ablate_tasks,
                       custom_interaction=custom_interaction, predictive=predictive,
                       subjects_factors=block_subjects_factors, use_mean=use_mean)

class DeepTFAModel(nn.Module):
    """Generative model for deep topographic factor analysis"""
    def __init__(self, locations, block_subjects, block_tasks, block_interactions,
                 num_factors=tfa_models.NUM_FACTORS, num_blocks=1,
                 num_times=[1], embedding_dim=2, voxel_noise=tfa_models.VOXEL_NOISE):
        super(self.__class__, self).__init__()
        self._locations = locations
        self._num_factors = num_factors
        self._num_blocks = num_blocks
        self._num_times = num_times
        self.register_buffer('block_subjects', torch.tensor(block_subjects,
                                                            dtype=torch.long),
                             persistent=False)
        self.register_buffer('block_tasks', torch.tensor(block_tasks,
                                                         dtype=torch.long),
                             persistent=False)
        self.register_buffer('block_interactions', torch.tensor(block_interactions,
                                                         dtype=torch.long),
                             persistent=False)

        self.hyperparams = DeepTFAGenerativeHyperparams(
            len(self.block_subjects.unique()), len(self.block_tasks.unique()), len(self.block_interactions.unique()),
            embedding_dim, voxel_noise=voxel_noise,
        )
        self.add_module('likelihood', tfa_models.TFAGenerativeLikelihood(
            locations, self._num_times, block=None, register_locations=False
        ))

    def forward(self, decoder, trace, times=None, guide=None, observations=[],
                blocks=None, locations=None, params=None,
                num_particles=tfa_models.NUM_PARTICLES, ablate_subjects=False, ablate_tasks=False,
                custom_interaction=None, predictive=False, block_subjects_factors=None):
        if params is None:
            params = self.hyperparams.state_vardict(num_particles)
        if guide is None:
            guide = probtorch.Trace()
        if times is None:
            times = torch.arange(max(self._num_times))
        if blocks is None:
            blocks = torch.arange(self._num_blocks)

        unique_blocks, block_idx = blocks.unique(return_inverse=True)

        block_subjects = self.block_subjects[unique_blocks]
        block_tasks = self.block_tasks[unique_blocks]
        block_interactions = self.block_interactions[unique_blocks]
        if block_subjects_factors is None:
            block_subjects_factors = block_subjects
        else:
            block_subjects_factors = self.block_subjects[block_subjects_factors.unique()]

        weights, centers, log_widths, participant_weight, stimulus_weight, interaction_weight \
            = decoder(trace, blocks, block_subjects, block_tasks, block_interactions,
                                               params, times,
                                               guide=guide, generative=True,
                                               ablate_subjects=ablate_subjects,
                                               ablate_tasks=ablate_tasks, custom_interaction=custom_interaction,
                                               subjects_factors=block_subjects_factors,
                      )
        return self.likelihood(trace, weights, centers, log_widths, params,
                               times=times, observations=observations,
                               block_idx=block_idx, locations=locations), \
               participant_weight, stimulus_weight, interaction_weight
