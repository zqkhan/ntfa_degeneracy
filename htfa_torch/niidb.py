"""Utilities for topographic factor analysis"""

__author__ = ('Jan-Willem van de Meent',
              'Eli Sennesh',
              'Zulqarnain Khan')
__email__ = ('j.vandemeent@northeastern.edu',
             'sennesh.e@husky.neu.edu',
             'khan.zu@husky.neu.edu')
from functools import lru_cache
import json
import logging
from ordered_set import OrderedSet
import types

import dataset
import torch.utils.data

from . import utils

@lru_cache(maxsize=16)
def lru_load_dataset(fname, mask, zscore, smooth,
                     zscore_by_rest=False,rest_starts=None,
                     rest_ends=None):
    logging.info('Loading Nifti image %s with mask %s (zscore=%s, smooth=%s, zscore_by_rest=%s)',
                 fname, mask, zscore, smooth,zscore_by_rest)
    return utils.load_dataset(fname, mask, smooth=smooth, zscore=zscore,
                              zscore_by_rest=zscore_by_rest, rest_starts=rest_starts,
                              rest_ends=rest_ends)

class FMriActivationBlock(object):
    def __init__(self, zscore=True, zscore_by_rest=False, smooth=None):
        self._zscore = zscore
        self._zscore_by_rest = zscore_by_rest
        self.smooth = smooth
        self.filename = ''
        self.mask = None
        self.subject = 0
        self.run = 0
        self.task = None
        self.block = 0
        self.start_time = None
        self.end_time = None
        self.rest_start_times = None
        self.rest_end_times = None
        self.activations = None
        self.locations = None
        self.individual_differences = {}

    def load(self):
        self.activations, self.locations, _, _ =\
            lru_load_dataset(self.filename, self.mask, self._zscore,
                             self.smooth, self._zscore_by_rest,
                             self.rest_start_times,self.rest_end_times)
        if self.start_time is None:
            self.start_time = 0
        if self.end_time is None:
            self.end_time = self.activations.shape[0]
        self.activations = self.activations[self.start_time:self.end_time]

    def unload(self):
        del self.activations
        del self.locations
        self.activations = None
        self.locations = None

    def unload_locations(self):
        del self.locations
        self.locations = None

    def __len__(self):
        return self.activations.shape[0]

    def default_label(self):
        return "subject%d_run%d_block%d" % (self.subject, self.run, self.block)

class FMriActivationsDb:
    def __init__(self, name, mask=None, smooth=None):
        self._db = dataset.connect('sqlite:///%s' % name)
        self._table = self._db['fmri_activations']
        self.mask = mask
        self.smooth = smooth

    def inference_filter(self, training=True, held_out_subjects=set(),
                         held_out_tasks=set()):
        subjects = OrderedSet([b.subject for b in self.all()])
        subjects = subjects - held_out_subjects
        tasks = OrderedSet([b.task for b in self.all()]) - held_out_tasks
        diagonals = list(utils.striping_diagonal_indices(len(subjects),
                                                         len(tasks)))
        def result(b):
            subject_index = subjects.index(b.subject)
            task_index = tasks.index(b.task)
            return ((subject_index, task_index) in diagonals) == (not training)

        return result

    def insert(self, block):
        if self.mask is not None:
            block.mask = self.mask
            block.smooth = self.smooth
        block_dict = block.__dict__.copy()
        del block_dict['activations']
        del block_dict['locations']
        self._table.insert(block_dict)

    def update(self, block, cols):
        block_dict = block.__dict__.copy()
        del block_dict['activations']
        del block_dict['locations']
        self._table.update(block_dict, cols)

    def upsert(self, block):
        if self.mask is not None:
            block.mask = self.mask
            block.smooth = self.smooth
        block_dict = block.__dict__.copy()
        del block_dict['activations']
        del block_dict['locations']
        block_dict['individual_differences'] =\
            json.dumps(block_dict['individual_differences'])
        block_dict['rest_start_times'] =\
            json.dumps(block_dict['rest_start_times'])
        block_dict['rest_end_times'] =\
            json.dumps(block_dict['rest_end_times'])
        self._table.upsert(block_dict, ['subject', 'run', 'task', 'block',
                                        'start_time', 'end_time',
                                        'rest_start_times', 'rest_end_times',
                                        'individual_differences'])

    def __getattr__(self, name):
        attr = getattr(self._table, name)
        if isinstance(attr, types.MethodType):
            def wrapped_table_method(*args, **kwargs):
                block_dicts = attr(*args, **kwargs)
                if hasattr(block_dicts, '__iter__') or\
                   hasattr(block_dicts, 'next'):
                    #block_dicts is an iterable or iterator
                    for block_dict in block_dicts:
                        block = FMriActivationBlock()
                        block.__dict__.update(**block_dict)
                        if block.individual_differences:
                            block.individual_differences =\
                                json.loads(block.individual_differences)
                        if self.mask is not None:
                            block.mask = self.mask
                        yield block
                elif isinstance(block_dicts, dict):
                    block = FMriActivationBlock()
                    block.__dict__.update(**block_dicts)
                    if self.mask is not None:
                        block.mask = self.mask
                    return block
                else:
                    return block_dicts
            return wrapped_table_method
        return attr

def query_max_time(qiter):
    return max([block.end_time - block.start_time for block in qiter])

def query_min_time(qiter):
    return min([block.end_time - block.start_time for block in qiter])

class QueryDataset(torch.utils.data.Dataset):
    def __init__(self, qiter):
        self.blocks = list(qiter)
        self._num_times = query_min_time(self.blocks)

    def __len__(self):
        return self._num_times

    def load(self):
        for block in self.blocks:
            block.load()

    def unload(self):
        for block in self.blocks:
            block.unload()

    def __getitem__(self, i):
        return torch.stack([block.activations[i] for block in self.blocks],
                           dim=0)
