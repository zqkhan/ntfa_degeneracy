#!/usr/bin/env python

import argparse
import logging

import htfa_torch.tfa as TFA
import htfa_torch.utils as utils

parser = argparse.ArgumentParser(description='Topographical factor analysis for fMRI data')
parser.add_argument('data_file', type=str, help='fMRI filename')
parser.add_argument('--steps', type=int, default=100, help='Number of optimization steps')
parser.add_argument('--learning_rate', type=float, default=TFA.LEARNING_RATE,
                    help='Learning Rate for optimization')
parser.add_argument('--log-optimization', action='store_true', help='Whether to log optimization')
parser.add_argument('--factors', type=int, default=TFA.NUM_FACTORS, help='Number of latent factors')

if __name__ == '__main__':
    args = parser.parse_args()
    tfa = TFA.TopographicalFactorAnalysis(args.data_file, num_factors=args.factors)
    if args.log_optimization:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    losses = tfa.train(num_steps=args.steps, learning_rate=args.learning_rate,
                       log_level=log_level)
    if args.log_optimization:
        utils.plot_losses(losses)
    tfa.mean_parameters(log_level=log_level)
