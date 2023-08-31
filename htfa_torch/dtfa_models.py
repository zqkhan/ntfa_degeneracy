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

TEMPLATE_SHAPE = utils.vardict({
    'factor_centers': None,
    'factor_log_widths': None,
})

class DeepTFAGenerativeHyperparams(tfa_models.HyperParams):
    def __init__(self, num_subjects, num_factors, brain_center, brain_center_std_dev, embedding_dim=2,
                 voxel_noise=tfa_models.VOXEL_NOISE, volume=None):
        self.num_subjects = num_subjects
        self.embedding_dim = embedding_dim
        self._num_factors = num_factors
        self.embedding_dim = embedding_dim

        params = utils.vardict()
        params['template'] = utils.populate_vardict(
            utils.vardict(TEMPLATE_SHAPE.copy()),
            utils.gaussian_populator,
            self._num_factors
        )

        coefficient = 1.0
        if volume is not None:
            coefficient = np.cbrt(volume / self._num_factors)
        params = utils.vardict({
            # 'subject': {
            #     'mu': torch.zeros(self.num_subjects, self.embedding_dim),
            #     'log_sigma': torch.ones(self.num_subjects, self.embedding_dim).log(),
            # },
            'template_factor_centers': {
                'mu': brain_center.expand(self._num_factors, 3),
                'log_sigma': torch.log(brain_center_std_dev / coefficient).expand(self._num_factors, 3),
            },
            'template_factor_log_widths': {
                'mu': torch.ones(self._num_factors) * np.log(coefficient),
                'log_sigma': torch.zeros(self._num_factors),
            },
            'subject_weight': {
                'mu': torch.zeros(self.num_subjects, self.embedding_dim),
                'log_sigma': torch.ones(self.num_subjects, self.embedding_dim).log(),
            },
            # 'task': {
            #     'mu': torch.zeros(self.num_tasks, self.embedding_dim),
            #     'log_sigma': torch.ones(self.num_tasks, self.embedding_dim).log(),
            # },
            # 'interaction': {
            #     'mu': torch.zeros(self.num_subjects * self.num_tasks, self.embedding_dim),
            #     'log_sigma': torch.ones(self.num_subjects * self.num_tasks, self.embedding_dim).log(),
            # },
            'voxel_noise': (torch.ones(1) * voxel_noise).log(), ##denominated in log_sigma
        })

        super(self.__class__, self).__init__(params, guide=False)

class DeepTFAGuideHyperparams(tfa_models.HyperParams):
    def __init__(self, num_blocks, num_times, num_factors, num_subjects, hyper_means, embedding_dim=2, time_series=True):
        self.num_blocks = num_blocks
        self.num_subjects = num_subjects
        self.num_times = max(num_times)
        self._num_factors = num_factors
        self.embedding_dim = embedding_dim


        params = utils.vardict({
            # 'subject': {
            #     'mu': torch.zeros(self.num_subjects, self.embedding_dim),
            #     'log_sigma': torch.ones(self.num_subjects, self.embedding_dim).log(),
            # },
            'template_factor_centers': {
                'mu': hyper_means['factor_centers'],
                'log_sigma': torch.zeros(self._num_factors, 3),
            },
            'template_factor_log_widths': {
                'mu': hyper_means['factor_log_widths'],
                'log_sigma': torch.zeros(self._num_factors),
            },
            'subject_weight': {
                'mu': torch.zeros(self.num_subjects, self.embedding_dim),
                'log_sigma': torch.ones(self.num_subjects, self.embedding_dim).log(),
            },
            'factor_centers': {
                'mu': hyper_means['factor_centers'].\
                        repeat(self.num_subjects, 1, 1),
                'log_sigma': torch.zeros(self.num_subjects, self._num_factors, 3),
            },
            'factor_log_widths': {
                'mu': torch.ones(self.num_subjects, self._num_factors) *\
                      hyper_means['factor_log_widths'],
                'log_sigma': torch.zeros(self.num_subjects, self._num_factors),
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
            if param == 'factor_centers' or param == 'factor_log_widths':
                mu = predictions[0]
                log_sigma = predictions[1]
            else:
                mu = predictions.select(-1, 0)
                log_sigma = predictions.select(-1, 1)
        else:
            mu = params[param]['mu']
            log_sigma = params[param]['log_sigma']
            if index is None:
                if param == 'template_factor_centers' or param == 'template_factor_log_widths':
                    mu = mu
                    log_sigma = log_sigma
                else:
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

    def forward(self, trace, blocks, subjects, params, times, guide=None,
                generative=False, predictive=False, subjects_factors=None, use_mean=False):
        origin = torch.zeros(params['subject_weight']['mu'].shape[0], len(blocks),
                             self._embedding_dim)
        origin = origin.to(params['subject_weight']['mu'])
        # if subjects_factors is None:
        #     subjects_factors = subjects
        if subjects is not None:
            # subject_embed = self._predict_param(
            #     params, 'subject', subjects_factors, None,
            #     'z^PF', trace, False, guide, use_mean=(use_mean) and not (generative),
            # )
            subject_weight_embed = self._predict_param(
                params, 'subject_weight', subjects, None,
                'z^PW', trace, False, guide, use_mean=(use_mean) and not (generative),
            )
        else:
            # subject_embed = origin
            subject_weight_embed = origin

        ### defining W = W_p + W_s + W_ps, which means adding their means and variances. the sqrt/logs/exp are because
        ### everything is defined in terms of log_sigmas.


        weight_predictions = (self.weights_embedding(subject_weight_embed)).view(
            subject_weight_embed.shape[0], subject_weight_embed.shape[1], self._num_factors, 2
        )

        weight_predictions = weight_predictions.unsqueeze(2).expand(
            subject_weight_embed.shape[0], subject_weight_embed.shape[1], len(times),
            self._num_factors, 2
        )

        template_centers_predictions = self._predict_param(
            params, 'template_factor_centers', None, None,
            'TemplateFactorCenters', trace, False, guide,
        )

        template_log_widths_predictions = self._predict_param(
            params, 'template_factor_log_widths', None, None,
            'TemplateFactorLogWidths', trace, False, guide,
        )
        if not generative:
            centers_predictions = self._predict_param(
                params, 'factor_centers', subjects, template_centers_predictions, #.unsqueeze(0),
                'FactorCenters', trace, predict=generative, guide=guide, use_mean=(use_mean) and not (generative),
            )

            log_widths_predictions = self._predict_param(
                params, 'factor_log_widths', subjects, template_log_widths_predictions, #.unsqueeze(0),
                'FactorLogWidths', trace, predict=generative, guide=guide, use_mean=(use_mean) and not (generative),
            )
        else:
            generative_template_centers_prediction = \
                [params['template_factor_centers']['mu'].expand(origin.shape[0], len(subjects), self._num_factors, 3),
                 params['template_factor_centers']['log_sigma'].expand(origin.shape[0],
                                                                       len(subjects), self._num_factors, 3)]
            centers_predictions = self._predict_param(
                params, 'factor_centers', subjects,
                generative_template_centers_prediction,  # .unsqueeze(0),
                'FactorCenters', trace, predict=generative, guide=guide, use_mean=(use_mean) and not (generative),
            )
            generative_template_log_widths_prediction = \
                [params['template_factor_log_widths']['mu'].expand(origin.shape[0], len(subjects), self._num_factors),
                 params['template_factor_log_widths']['log_sigma'].expand(origin.shape[0],
                                                                          len(subjects), self._num_factors)]

            log_widths_predictions = self._predict_param(
                params, 'factor_log_widths', subjects, generative_template_log_widths_prediction,  # .unsqueeze(0),
                'FactorLogWidths', trace, predict=generative, guide=guide, use_mean=(use_mean) and not (generative),
            )
        if 'locations_min' in self._buffers:
            centers_predictions = utils.clamp_locations(centers_predictions,
                                                        self.locations_min,
                                                        self.locations_max)
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

        return weight_predictions, centers_predictions, log_widths_predictions,

class DeepTFAGuide(nn.Module):
    """Variational guide for deep topographic factor analysis"""
    def __init__(self, num_factors, block_subjects, num_blocks=1,
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

        num_subjects = len(self.block_subjects.unique())


        self.hyperparams = DeepTFAGuideHyperparams(self._num_blocks,
                                                   self._num_times,
                                                   self._num_factors,
                                                   num_subjects,
                                                   hyper_means,
                                                   embedding_dim, time_series)

    def forward(self, decoder, trace, times=None, blocks=None, params=None,
                num_particles=tfa_models.NUM_PARTICLES, predictive=False, use_mean=False):
        if params is None:
            params = self.hyperparams.state_vardict(num_particles)
        if blocks is None:
            blocks = torch.arange(self._num_blocks)

        unique_blocks = blocks.unique()
        block_subjects = self.block_subjects[unique_blocks]

        return decoder(trace, blocks, block_subjects, params,
                       times=times, predictive=predictive, use_mean=use_mean)

class DeepTFAModel(nn.Module):
    """Generative model for deep topographic factor analysis"""
    def __init__(self, locations, block_subjects,
                 num_factors=tfa_models.NUM_FACTORS, num_blocks=1,
                 num_times=[1], embedding_dim=2, voxel_noise=tfa_models.VOXEL_NOISE, volume=None):
        super(self.__class__, self).__init__()
        self._locations = locations
        self._num_factors = num_factors
        self._num_blocks = num_blocks
        self._num_times = num_times
        self.register_buffer('block_subjects', torch.tensor(block_subjects,
                                                            dtype=torch.long),
                             persistent=False)

        center, center_sigma = utils.brain_centroid(locations)
        hull = scipy.spatial.ConvexHull(locations)
        if volume is not None:
            volume = hull.volume

        self.hyperparams = DeepTFAGenerativeHyperparams(
            len(self.block_subjects.unique()), self._num_factors, center, center_sigma,
            embedding_dim, voxel_noise=voxel_noise, volume=volume
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
        # block_tasks = self.block_tasks[unique_blocks]
        # block_interactions = self.block_interactions[unique_blocks]

        weights, centers, log_widths = decoder(trace, blocks, block_subjects,
                       params, times,
                       guide=guide, generative=True,
                      )
        return self.likelihood(trace, weights, centers, log_widths, params,
                               times=times, observations=observations,
                               block_idx=block_idx, locations=locations)
