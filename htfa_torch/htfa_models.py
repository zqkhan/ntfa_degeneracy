"""Hierarchical factor analysis models as ProbTorch modules"""

__author__ = 'Eli Sennesh', 'Zulqarnain Khan'
__email__ = 'e.sennesh@northeastern.edu', 'khan.zu@husky.neu.edu'

import collections
import numpy as np
import scipy.spatial
import torch
from torch.autograd import Variable
import probtorch
import torch.nn as nn
from torch.nn.functional import softplus

from . import niidb
from . import tfa_models
from . import utils

TEMPLATE_SHAPE = utils.vardict({
    'factor_centers': None,
    'factor_log_widths': None,
})

class HTFAGuideHyperParams(tfa_models.HyperParams):
    def __init__(self, hyper_means, num_times, num_blocks,
                 num_factors=tfa_models.NUM_FACTORS):
        self._num_times = num_times
        self._num_blocks = num_blocks
        self._num_factors = num_factors

        params = utils.vardict()
        params['template'] = utils.populate_vardict(
            utils.vardict(TEMPLATE_SHAPE.copy()),
            utils.gaussian_populator,
            self._num_factors
        )
        params['template']['factor_centers'] =\
            utils.gaussian_populator(self._num_factors, 3)
        params['template']['factor_centers']['mu'] +=\
            hyper_means['factor_centers']
        params['template']['factor_centers']['log_sigma'] =\
            torch.zeros(self._num_factors, 3)
        params['template']['factor_log_widths']['mu'] =\
            hyper_means['factor_log_widths']
        params['template']['factor_log_widths']['log_sigma'] =\
            torch.zeros(self._num_factors)

        params['block'] = utils.vardict({
            'factor_centers': {
                'mu': hyper_means['factor_centers'].\
                        repeat(self._num_blocks, 1, 1),
                'log_sigma': torch.zeros(self._num_blocks, self._num_factors, 3),
            },
            'factor_log_widths': {
                'mu': torch.ones(self._num_blocks, self._num_factors) *\
                      hyper_means['factor_log_widths'],
                'log_sigma': torch.zeros(self._num_blocks, self._num_factors),
            },
            'weights': {
                'mu':  hyper_means['weights'].mean(0).unsqueeze(0).expand(
                    self._num_blocks, self._num_times, self._num_factors
                ),
                'log_sigma': torch.zeros(self._num_blocks, self._num_times,
                                         self._num_factors),
            },
        })

        super(self.__class__, self).__init__(params, guide=True)

class HTFAGuideTemplatePrior(tfa_models.GuidePrior):
    def forward(self, trace, params, template_shape=TEMPLATE_SHAPE,
                num_particles=tfa_models.NUM_PARTICLES):
        template_params = params['template']
        if num_particles and num_particles > 0:
            template_params = utils.unsqueeze_and_expand_vardict(
                template_params, 0, num_particles, True
            )
        template = template_shape.copy()
        for (k, _) in template.iteritems():
            if 'template_' + k in trace:
                continue
            template[k] = trace.normal(
                template_params[k]['mu'],
                torch.exp(template_params[k]['log_sigma']),
                name='template_' + k
            )

        return template

class HTFAGuideSubjectPrior(tfa_models.GuidePrior):
    def __init__(self, num_blocks, num_times):
        super(self.__class__, self).__init__()
        self._num_blocks = num_blocks
        self._num_times = num_times
        self._tfa_priors = [tfa_models.TFAGuidePrior(block=b)\
                            for b in range(self._num_blocks)]

    def forward(self, trace, params, times=None, blocks=None,
                num_particles=tfa_models.NUM_PARTICLES):
        # We only expand the parameters for which we're actually going to sample
        # values in this very method, and thus want to expand to get multiple
        # particles.
        if blocks is None:
            blocks = list(range(self._num_blocks))

        weights = []
        factor_centers = []
        factor_log_widths = []
        scan_times = times is None
        for b in blocks:
            if scan_times:
                times = (0, self._num_times[b])
            # The TFA prior is going to expand out particles all on its own, so
            # we never actually have to expand them.
            sparams = utils.vardict(params['block'])
            for k, v in sparams.iteritems():
                sparams[k] = v[b]
            w, fc, flw = self._tfa_priors[b](trace, sparams, times=times,
                                             num_particles=num_particles)
            weights += [w]
            factor_centers += [fc]
            factor_log_widths += [flw]

        return weights, factor_centers, factor_log_widths

class HTFAGuide(nn.Module):
    """Variational guide for hierarchical topographic factor analysis"""
    def __init__(self, query, num_factors=tfa_models.NUM_FACTORS):
        super(self.__class__, self).__init__()
        self._num_blocks = len(query)
        self._num_times = niidb.query_max_time(query)

        b = max(range(self._num_blocks), key=lambda b: query[b].end_time -
                query[b].start_time)
        query[b].load()
        centers, widths, weights = utils.initial_hypermeans(
            query[b].activations.numpy().T, query[b].locations.numpy(),
            num_factors
        )
        hyper_means = {
            'weights': torch.Tensor(weights),
            'factor_centers': torch.Tensor(centers),
            'factor_log_widths': widths,
        }
        self.hyperparams = HTFAGuideHyperParams(hyper_means,
                                                self._num_times,
                                                self._num_blocks, num_factors)
        self._template_prior = HTFAGuideTemplatePrior()
        self._subject_prior = HTFAGuideSubjectPrior(self._num_blocks,
                                                    self._num_times)

    def forward(self, trace, times=None, blocks=None,
                num_particles=tfa_models.NUM_PARTICLES):
        params = self.hyperparams.state_vardict()
        self._template_prior(trace, params, num_particles=num_particles)
        return self._subject_prior(trace, params, times=times, blocks=blocks,
                                   num_particles=num_particles)

class HTFAGenerativeHyperParams(tfa_models.HyperParams):
    def __init__(self, brain_center, brain_center_std_dev, num_blocks,
                 num_factors=tfa_models.NUM_FACTORS, volume=None):
        self._num_factors = num_factors
        self._num_blocks = num_blocks

        params = utils.vardict()
        params['template'] = utils.populate_vardict(
            utils.vardict(TEMPLATE_SHAPE.copy()),
            utils.gaussian_populator,
            self._num_factors
        )

        coefficient = 1.0
        if volume is not None:
            coefficient = np.cbrt(volume / self._num_factors)

        params['template']['factor_centers']['mu'] =\
            brain_center.expand(self._num_factors, 3)
        params['template']['factor_centers']['log_sigma'] =\
            torch.log(brain_center_std_dev / coefficient)

        params['template']['factor_log_widths']['mu'] =\
            torch.ones(self._num_factors) * np.log(coefficient)
        params['template']['factor_log_widths']['log_sigma'] =\
            torch.zeros(self._num_factors)

        params['weights'] = {
            'mu': torch.zeros(self._num_factors),
            'log_sigma': torch.zeros(self._num_factors),
        }

        params['voxel_noise'] = torch.ones(1) * tfa_models.VOXEL_NOISE

        super(self.__class__, self).__init__(params, guide=False)

class HTFAGenerativeTemplatePrior(tfa_models.GenerativePrior):
    def forward(self, trace, params, template_shape=TEMPLATE_SHAPE,
                guide=probtorch.Trace()):
        template = utils.vardict(template_shape.copy())
        for (k, _) in template.iteritems():
            if 'template_' + k in trace:
                continue                
            template[k] = trace.normal(
                    params['template'][k]['mu'],
                    torch.exp(params['template'][k]['log_sigma']),
                    value=guide['template_' + k], name='template_' + k
                )
        return template

class HTFAGenerativeSubjectPrior(tfa_models.GenerativePrior):
    def __init__(self, num_blocks, num_times):
        super(self.__class__, self).__init__()
        self._num_blocks = num_blocks
        self._num_times = num_times
        self._tfa_priors = [tfa_models.TFAGenerativePrior(self._num_times[b],
                                                          block=b)\
                            for b in range(self._num_blocks)]

    def forward(self, trace, params, template, times=None, blocks=None,
                guide=probtorch.Trace(), weights_params=None):
        if blocks is None:
            blocks = list(range(self._num_blocks))
        if times is None:
            times = (0, self._num_times)

        weights = []
        factor_centers = []
        factor_log_widths = []
        for (i, b) in enumerate(blocks):
            sparams = utils.vardict({
                'factor_centers': {
                    'mu': template['factor_centers'],
                    'log_sigma': torch.zeros(
                        template['factor_centers'].shape[0], 3
                    ).to(template['factor_centers']),
                },
                'factor_log_widths': {
                    'mu': template['factor_log_widths'],
                    'log_sigma': torch.zeros(template['factor_log_widths'].shape).to(
                        template['factor_log_widths']
                    ),
                },
                'weights': {
                    'mu': params['weights']['mu'],
                    'log_sigma': params['weights']['log_sigma'],
                },
            })
            if weights_params is not None:
                sparams['weights'] = weights_params[i]
            w, fc, flw = self._tfa_priors[b](trace, sparams, times=times,
                                             guide=guide)
            weights += [w]
            factor_centers += [fc]
            factor_log_widths += [flw]

        return weights, factor_centers, factor_log_widths

class HTFAModel(nn.Module):
    """Generative model for hierarchical topographic factor analysis"""
    def __init__(self, locations, num_blocks, num_times,
                 num_factors=tfa_models.NUM_FACTORS, volume=None):
        super(self.__class__, self).__init__()

        self._num_factors = num_factors
        self._num_blocks = num_blocks
        self._num_times = num_times

        center, center_sigma = utils.brain_centroid(locations)
        hull = scipy.spatial.ConvexHull(locations)
        if volume is not None:
            volume = hull.volume

        self._hyperparams = HTFAGenerativeHyperParams(center, center_sigma,
                                                      self._num_blocks,
                                                      self._num_factors,
                                                      volume=volume)
        self._template_prior = HTFAGenerativeTemplatePrior()
        self._subject_prior = HTFAGenerativeSubjectPrior(
            self._num_blocks, self._num_times
        )
        self.likelihoods = [tfa_models.TFAGenerativeLikelihood(
            locations, self._num_times[b], block=b, register_locations=False
        ) for b in range(self._num_blocks)]
        for b, block_likelihood in enumerate(self.likelihoods):
            self.add_module('likelihood' + str(b), block_likelihood)

    def forward(self, trace, times=None, guide=probtorch.Trace(), blocks=None,
                observations=[], weights_params=None):
        if blocks is None:
            blocks = list(range(self._num_blocks))
        params = self._hyperparams.state_vardict()

        template = self._template_prior(trace, params, guide=guide)
        weights, centers, log_widths = self._subject_prior(
            trace, params, template, times=times, blocks=blocks, guide=guide,
            weights_params=weights_params
        )

        return [self.likelihoods[b](trace, weights[i], centers[i],
                                    log_widths[i], params, times=times,
                                    observations=observations[i])
                for (i, b) in enumerate(blocks)]
