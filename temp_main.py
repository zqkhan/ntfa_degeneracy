import logging
import numpy as np
import htfa_torch.dtfa as DTFA
import htfa_torch.niidb as niidb
import htfa_torch.utils as utils
import matplotlib.pyplot as plt
from ordered_set import OrderedSet
import os
from torch.nn.functional import softplus
import torch
import itertools
from htfa_torch import tfa_models
import nilearn.plotting as niplot
import imageio
from htfa_torch import tardb
import glob

def getEquidistantPoints(p1, p2, parts):
    return zip(np.linspace(p1[0], p2[0], parts + 1), np.linspace(p1[1], p2[1], parts + 1))


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
# AVFP_FILE = '/home/zulqarnain/Code/ntfa_v2_nofactorembed/ntfa_degeneracy/data/traumarecall_shards-*.tar'
# shard_list = sorted(
#     glob.glob(AVFP_FILE)
# )
#
# meta_path = '/home/zulqarnain/Code/ntfa_v2_nofactorembed/ntfa_degeneracy/data/traumarecall_shards.tar.meta'
# avfp_db = tardb.FmriTarDataset(shard_list, selector=lambda block: True, meta_path=meta_path)

AVFP_FILE = "/home/zulqarnain/Code/ntfa_v2_nofactorembed/ntfa_degeneracy/data/traumarecall_filmreact.tar"
avfp_db = tardb.FmriTarDataset(AVFP_FILE)
dtfa = DTFA.DeepTFA(avfp_db, num_factors=100)
losses = dtfa.train(num_steps=10, learning_rate={'q': 1e-2, 'p': 1e-4}, log_level=logging.INFO, num_particles=1,
                    batch_size=320, use_cuda=True, checkpoint_steps=500, patience=50)
# dtfa.load_state('participant_CHECK_01062020_134125') #for scenario 1
# dtfa.load_state('participant_CHECK_01062020_155320')  # for scenario 0


