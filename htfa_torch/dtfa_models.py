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
    def __init__(self, num_subjects, num_tasks, embedding_dim=2):
        self.num_subjects = num_subjects
        self.num_tasks = num_tasks
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

            'voxel_noise': torch.ones(1) * tfa_models.VOXEL_NOISE,
        })

        super(self.__class__, self).__init__(params, guide=False)

class DeepTFAGuideHyperparams(tfa_models.HyperParams):
    def __init__(self, num_blocks, num_times, num_factors, num_subjects,
                 num_tasks, hyper_means, embedding_dim=2, time_series=True):
        self.num_blocks = num_blocks
        self.num_subjects = num_subjects
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
                 linear='PSC'):
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
        
        self.interaction_embedding = nn.Sequential(
            nn.Linear(self._embedding_dim * 2, self._embedding_dim * 4, bias=False),
            nn.PReLU(),
            nn.Linear(self._embedding_dim * 4, self._embedding_dim, bias=False)
        )  
        
        if 'P' in linear:
            self.participant_weights_embedding = nn.Sequential(
                nn.Linear(self._embedding_dim, self._num_factors, bias=False)
            )
        else:
            self.participant_weights_embedding = nn.Sequential(
                nn.Linear(self._embedding_dim, self._embedding_dim * 4, bias=False),
                nn.PReLU(),
                nn.Linear(self._embedding_dim * 4, self._num_factors, bias=False)
            )
        if 'S' in linear:
            self.stimulus_weights_embedding = nn.Sequential(
                nn.Linear(self._embedding_dim, self._num_factors, bias=False)
            )
        else:
            self.stimulus_weights_embedding = nn.Sequential(
                nn.Linear(self._embedding_dim, self._embedding_dim * 4, bias=False),
                nn.PReLU(),
                nn.Linear(self._embedding_dim * 4, self._num_factors, bias=False)
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
                       predict=True, guide=None):
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

        result = trace.normal(mu, torch.exp(log_sigma),
                              value=utils.clamped(name, guide), name=name)
        return result

    def forward(self, trace, blocks, subjects, tasks, params, times, guide=None,
                generative=False):
        origin = torch.zeros(params['subject']['mu'].shape[0], len(blocks),
                             self._embedding_dim)
        origin = origin.to(params['subject']['mu'])
        if subjects is not None:
            subject_embed = self._predict_param(
                params, 'subject', subjects, None,
                'z^PF', trace, False, guide
            )
            subject_weight_embed = self._predict_param(
                params, 'subject_weight', subjects, None,
                'z^PW', trace, False, guide
            )
        else:
            subject_embed = origin
            subject_weight_embed = origin
        if tasks is not None:
            task_embed = self._predict_param(params, 'task', tasks, None, 'z^S',
                                             trace, False, guide)
        else:
            task_embed = origin
        joint_embed = torch.cat((subject_weight_embed, task_embed), dim=-1)
        interaction_embed = self.interaction_embedding(joint_embed)
        # interaction_embed = self.interaction_embedding_out(torch.cat((interaction_embed,joint_embed),
        #                                                              dim=-1))
        # factor_params = self.factors_embedding(subject_embed).view(
        #     -1, self._num_factors, 4, 2
        # )
        factor_params = (self.factors_embedding(subject_embed)).view(
            -1, self._num_factors, 4, 2
        )
        centers_predictions = factor_params[:, :, :3]
        log_widths_predictions = factor_params[:, :, 3]

        participant_weight_predictions = self.participant_weights_embedding(subject_weight_embed)
        stimulus_weight_predictions = self.stimulus_weights_embedding(task_embed)
        interaction_weight_predictions = self.weights_embedding(interaction_embed)\
            .view(-1, self._num_factors, 2)[:, :, 0]
        weight_predictions = (self.weights_embedding(interaction_embed)).view(
            interaction_embed.shape[0], interaction_embed.shape[1], self._num_factors, 2
        )
        weight_predictions[:, :, :, 0] = participant_weight_predictions + \
                                         stimulus_weight_predictions + \
                                         interaction_weight_predictions
        weight_predictions = weight_predictions.unsqueeze(2).expand(
            interaction_embed.shape[0], interaction_embed.shape[1], len(times),
            self._num_factors, 2
        )

        centers_predictions = self._predict_param(
            params, 'factor_centers', subjects, centers_predictions,
            'FactorCenters', trace, predict=generative, guide=guide,
        )
        if 'locations_min' in self._buffers:
            centers_predictions = utils.clamp_locations(centers_predictions,
                                                        self.locations_min,
                                                        self.locations_max)
        log_widths_predictions = self._predict_param(
            params, 'factor_log_widths', subjects, log_widths_predictions,
            'FactorLogWidths', trace, predict=generative, guide=guide,
        )

        if generative:
            _, block_indices = blocks.unique(return_inverse=True)
            time_idx = torch.arange(len(times), dtype=torch.long)
            weight_predictions = weight_predictions[:, block_indices, time_idx]
        weight_predictions = self._predict_param(
            params, 'weights', (blocks, times), weight_predictions,
            'Weights_%s' % [t.item() for t in times], trace,
            predict=generative or (blocks < 0).any() or not self._time_series,
            guide=guide,
        )

        return weight_predictions, centers_predictions, log_widths_predictions, \
               participant_weight_predictions, stimulus_weight_predictions, interaction_weight_predictions

class DeepTFAGuide(nn.Module):
    """Variational guide for deep topographic factor analysis"""
    def __init__(self, num_factors, block_subjects, block_tasks, num_blocks=1,
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
        num_subjects = len(self.block_subjects.unique())
        num_tasks = len(self.block_tasks.unique())

        self.hyperparams = DeepTFAGuideHyperparams(self._num_blocks,
                                                   self._num_times,
                                                   self._num_factors,
                                                   num_subjects, num_tasks,
                                                   hyper_means,
                                                   embedding_dim, time_series)

    def forward(self, decoder, trace, times=None, blocks=None, params=None,
                num_particles=tfa_models.NUM_PARTICLES):
        if params is None:
            params = self.hyperparams.state_vardict(num_particles)
        if blocks is None:
            blocks = torch.arange(self._num_blocks)

        unique_blocks = blocks.unique()
        block_subjects = self.block_subjects[unique_blocks]
        block_tasks = self.block_tasks[unique_blocks]

        return decoder(trace, blocks, block_subjects, block_tasks, params,
                       times=times)

class DeepTFAModel(nn.Module):
    """Generative model for deep topographic factor analysis"""
    def __init__(self, locations, block_subjects, block_tasks,
                 num_factors=tfa_models.NUM_FACTORS, num_blocks=1,
                 num_times=[1], embedding_dim=2):
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

        self.hyperparams = DeepTFAGenerativeHyperparams(
            len(self.block_subjects.unique()), len(self.block_tasks.unique()),
            embedding_dim
        )
        self.add_module('likelihood', tfa_models.TFAGenerativeLikelihood(
            locations, self._num_times, block=None, register_locations=False
        ))

    def forward(self, decoder, trace, times=None, guide=None, observations=[],
                blocks=None, locations=None, params=None,
                num_particles=tfa_models.NUM_PARTICLES):
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

        weights, centers, log_widths, participant_weight, stimulus_weight, interaction_weight \
            = decoder(trace, blocks, block_subjects,
                                               block_tasks, params, times,
                                               guide=guide, generative=True)

        return self.likelihood(trace, weights, centers, log_widths, params,
                               times=times, observations=observations,
                               block_idx=block_idx, locations=locations), \
               participant_weight, stimulus_weight, interaction_weight
