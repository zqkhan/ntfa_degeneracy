"""Topographic factor analysis models as ProbTorch modules"""

__author__ = 'Eli Sennesh', 'Zulqarnain Khan'
__email__ = 'e.sennesh@northeastern.edu', 'khan.zu@husky.neu.edu'

import collections
from functools import lru_cache

import numpy as np
import torch
import torch.distributions as dists
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.functional import softplus
import torch.utils.data

import probtorch

from . import utils

NUM_FACTORS = 5
NUM_PARTICLES = 10
SOURCE_CENTER_STD_DEV = np.sqrt(10)
SOURCE_WEIGHT_STD_DEV = np.sqrt(2.0)
SOURCE_LOG_WIDTH_STD_DEV = np.sqrt(3.0)
VOXEL_NOISE = 0.1

# locations: V x 3
# centers: S x K x 3
# log_widths: S x K
def radial_basis(locations, centers, log_widths):
    """The radial basis function used as the shape for the factors"""
    # V x 3 -> 1 x V x 3
    locations = locations.unsqueeze(0)
    if len(centers.shape) > 3:
        # 1 x V x 3 -> 1 x 1 x V x 3
        locations = locations.unsqueeze(0)
    # S x K x 3 -> S x K x 1 x 3
    centers = centers.unsqueeze(len(centers.shape) - 1)
    # S x K x V x 3
    delta2s = ((locations - centers)**2).sum(len(centers.shape) - 1)
    # S x K  -> S x K x 1
    log_widths = log_widths.unsqueeze(len(log_widths.shape))
    return torch.exp(-torch.exp(torch.log(delta2s) - log_widths))

class Model(nn.Module):
    def __init__(self):
        super(nn.Module, self).__init__()

    def forward(self, *args, trace=probtorch.Trace()):
        pass

class HyperParams(Model):
    def __init__(self, vs, guide=True):
        super(Model, self).__init__()

        self._guide = guide
        utils.register_vardict(vs, self, self._guide)

    def state_vardict(self, num_particles=None):
        result = utils.vardict(self.state_dict(keep_vars=True))
        for k, v in result.items():
            if num_particles:
                v = v.expand(num_particles, *v.shape)
            if not isinstance(v, Variable):
                v = Variable(v)
            result[k] = v

        return result

class GuidePrior(Model):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, trace, *args, num_particles=NUM_PARTICLES):
        pass

class GenerativePrior(Model):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, trace, *args, guide=probtorch.Trace()):
        pass

class GenerativeLikelihood(Model):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, trace, *args, observations=collections.defaultdict()):
        pass

class TFAGuideHyperParams(HyperParams):
    def __init__(self, means, num_times, num_factors=NUM_FACTORS):
        self._num_times = num_times
        self._num_factors = num_factors

        params = utils.vardict()
        params['weights'] = {
            'mu': means['weights'],
            'log_sigma': torch.sqrt(torch.rand(
                (self._num_times, self._num_factors)
            )).log()
        }
        params['factor_centers'] = {
            'mu': means['factor_centers'],
            'log_sigma': torch.sqrt(torch.rand((self._num_factors, 3))).log()
        }
        params['factor_log_widths'] = {
            'mu': means['factor_log_widths'] * torch.ones(self._num_factors),
            'log_sigma': torch.sqrt(torch.rand((self._num_factors))).log()
        }
        super(self.__class__, self).__init__(params, guide=True)

    def forward(self):
        return utils.vardict(super(self.__class__, self).forward())

class TFAGuidePrior(GuidePrior):
    def __init__(self, block=0):
        super(TFAGuidePrior, self).__init__()
        self.block = block

    def forward(self, trace, params, times=None, num_particles=NUM_PARTICLES):
        if times is None:
            times = (0, params['weights']['mu'].shape[0])

        weight_params = {
            'mu': params['weights']['mu'][times[0]:times[1], :],
            'log_sigma': params['weights']['log_sigma'][times[0]:times[1], :]
        }

        if num_particles and num_particles > 0:
            params = utils.unsqueeze_and_expand_vardict(params, 0, num_particles,
                                                        True)
            weight_params = utils.unsqueeze_and_expand_vardict(weight_params,
                                                               0,
                                                               num_particles,
                                                               True)

        weights = trace.normal(weight_params['mu'],
                               torch.exp(weight_params['log_sigma']),
                               name='Weights%dt%d-%d' % (self.block, times[0], times[1]))

        centers = trace.normal(params['factor_centers']['mu'],
                               torch.exp(params['factor_centers']['log_sigma']),
                               name='FactorCenters' + str(self.block))
        log_widths = trace.normal(
            params['factor_log_widths']['mu'],
            torch.exp(params['factor_log_widths']['log_sigma']),
            name='FactorLogWidths' + str(self.block)
        )
        return weights, centers, log_widths

class TFAGuide(nn.Module):
    """Variational guide for topographic factor analysis"""
    def __init__(self, means, num_times, num_factors=NUM_FACTORS, block=0):
        super(self.__class__, self).__init__()
        self.block = block

        self.hyperparams = TFAGuideHyperParams(means, num_times, num_factors)
        self._prior = TFAGuidePrior(block=block)

    def forward(self, trace, times=None, num_particles=NUM_PARTICLES):
        params = self.hyperparams.state_vardict()
        return self._prior(trace, params, times=times, num_particles=num_particles)

class TFAGenerativeHyperParams(HyperParams):
    def __init__(self, brain_center, brain_center_std_dev,
                 num_factors=NUM_FACTORS, voxel_noise=VOXEL_NOISE):
        self._num_factors = num_factors

        params = utils.vardict()
        params['weights'] = {
            'mu': torch.zeros((self._num_factors)),
            'log_sigma': (SOURCE_WEIGHT_STD_DEV * torch.ones((self._num_factors))).log()
        }
        params['factor_centers'] = {
            'mu': brain_center.expand(self._num_factors, 3) *\
                torch.ones((self._num_factors, 3)),
            'log_sigma': torch.log(brain_center_std_dev * SOURCE_CENTER_STD_DEV)
        }
        params['factor_log_widths'] = {
            'mu': torch.ones((self._num_factors)),
            'log_sigma': (SOURCE_LOG_WIDTH_STD_DEV * torch.ones((self._num_factors))).log()
        }
        params['voxel_noise'] = torch.ones(1) * voxel_noise

        super(self.__class__, self).__init__(params, guide=False)

    def forward(self):
        return utils.vardict(super(self.__class__, self).forward())

class TFAGenerativePrior(GenerativePrior):
    def __init__(self, num_times, block=0):
        super(self.__class__, self).__init__()
        self._num_times = num_times
        self.block = block

    def forward(self, trace, params, times=None, guide=probtorch.Trace()):
        if times is None:
            times = (0, self._num_times)

        weight_params = utils.unsqueeze_and_expand_vardict(
            params['weights'], len(params['weights']['mu'].shape) - 1,
            times[1] - times[0], True
        )

        weights = trace.normal(weight_params['mu'],
                               torch.exp(weight_params['log_sigma']),
                               value=guide['Weights%dt%d-%d' % (self.block, times[0], times[1])],
                               name='Weights%dt%d-%d' % (self.block, times[0], times[1]))

        factor_centers = trace.normal(
            params['factor_centers']['mu'], torch.exp(params['factor_centers']['log_sigma']),
            value=guide['FactorCenters' + str(self.block)],
            name='FactorCenters' + str(self.block)
        )
        factor_log_widths = trace.normal(params['factor_log_widths']['mu'],
                                         torch.exp(params['factor_log_widths']['log_sigma']),
                                         value=guide['FactorLogWidths' + str(self.block)],
                                         name='FactorLogWidths' + str(self.block))

        return weights, factor_centers, factor_log_widths

class TFAGenerativeLikelihood(GenerativeLikelihood):
    def __init__(self, locations, num_times, block=0,
                 register_locations=True):
        super(self.__class__, self).__init__()

        if register_locations:
            self.register_buffer('voxel_locations', locations)
        else:
            self.voxel_locations = locations
        self._num_times = num_times
        self.block = block

    def forward(self, trace, weights, centers, log_widths, params, times=None,
                observations=None, block_idx=None, locations=None):
        if times is None:
            times = torch.arange(self._num_times)
        if observations is None:
            observations = collections.defaultdict()
        if block_idx is None:
            blocks = torch.tensor([self.block], dtype=torch.long,
                                  device=weights.device)
            block_idx = blocks.unique(return_inverse=True)
        if locations is None:
            locations = self.voxel_locations

        time_idx = torch.arange(times.shape[0], device=locations.device)

        factors = radial_basis(locations, centers, log_widths)
        predictions = (weights @ factors)[:, block_idx, time_idx]

        activations = trace.normal(predictions, params['voxel_noise'][0],
                                   value=observations['Y'], name='Y')
        return activations

class TFAModel(nn.Module):
    """Generative model for topographic factor analysis"""
    def __init__(self, brain_center, brain_center_std_dev, num_times,
                 locations, num_factors=NUM_FACTORS, voxel_noise=VOXEL_NOISE,
                 block=0, register_locations=True):
        super(self.__class__, self).__init__()

        self._num_times = num_times
        self._num_factors = num_factors
        self._locations = locations
        self.block = block

        self._hyperparams = TFAGenerativeHyperParams(brain_center,
                                                     brain_center_std_dev,
                                                     self._num_factors,
                                                     voxel_noise)
        self._prior = TFAGenerativePrior(self._num_times, block=self.block)
        self._likelihood = TFAGenerativeLikelihood(self._locations,
                                                   self._num_times,
                                                   block=self.block,
                                                   register_locations=register_locations)

    def forward(self, trace, times=None, guide=probtorch.Trace(),
                observations=collections.defaultdict()):
        params = self._hyperparams.state_vardict()
        weights, centers, log_widths = self._prior(trace, params, times=times,
                                                   guide=guide)
        return self._likelihood(trace, weights, centers, log_widths,
                                times=times, observations=observations)
