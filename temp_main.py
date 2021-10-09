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


def getEquidistantPoints(p1, p2, parts):
    return zip(np.linspace(p1[0], p2[0], parts + 1), np.linspace(p1[1], p2[1], parts + 1))


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

synthetic_db = niidb.FMriActivationsDb('/home/zulqarnain/algorithm4/htfatorch/data/simulated_simplified_data_3_tiny.db')

dtfa = DTFA.DeepTFA(synthetic_db.all(), mask='/home/zulqarnain/fmri_data/degeneracy_scenario_1_data/wholebrain.nii.gz',
                    num_factors=9, embedding_dim=2)
losses = dtfa.train(num_steps=2, learning_rate={'q': 1e-2, 'p': 1e-4}, log_level=logging.INFO, num_particles=1,
                    batch_size=60, use_cuda=True, checkpoint_steps=500, blocks_batch_size=20, patience=50)
# dtfa.load_state('participant_CHECK_01062020_134125') #for scenario 1
# dtfa.load_state('participant_CHECK_01062020_155320')  # for scenario 0


def task_labeler(task):
    if task == 'heights_high':
        return 'h_h'
    elif task == 'heights_low':
        return 'h_l'
    elif task == 'social_high':
        return 'so_h'
    elif task == 'social_low':
        return 'so_l'
    elif task == 'spider_high':
        return 'sp_h'
    elif task == 'spider_low':
        return 'sp_l'
    else:
        return 'Other'


def subject_labeler(subject):
    return str(subject)


hyperparams = dtfa.variational.hyperparams.state_vardict()
z_i_mu = hyperparams['interactions']['mu'].data
z_pf_mu = hyperparams['subject']['mu'].data
z_pw_mu = hyperparams['subject_weight']['mu'].data
z_s_mu = hyperparams['task_weight']['mu'].data

tasks = dtfa.stimuli()
task_category = dtfa.tasks()
subjects = dtfa.subjects()
interactions = OrderedSet(list(itertools.product(subjects, tasks)))

i_count = 0
for p in range(len(subjects)):
    for s in range(len(tasks)):
        temp_i_mu = z_i_mu[i_count, :]
        i_count += 1
        temp_pf_mu = z_pf_mu[p, :]
        temp_pw_mu = z_pw_mu[p, :]
        for t in range(len(task_category)):
            if task_category[t] in tasks[s]:
                temp_s_mu = z_s_mu[t, :]
        joint_embed = torch.cat((temp_pw_mu, temp_s_mu, temp_i_mu), dim=-1)

        weight_predictions = dtfa.decoder.weights_embedding(joint_embed).view(
            -1, dtfa.num_factors, 2
        )
        mean_weight = weight_predictions[:, :, 0]
        factor_params = dtfa.decoder.factors_embedding(temp_pf_mu).view(
            -1, dtfa.num_factors, 4, 2
        )
        centers_predictions = factor_params[:, :, :3]
        log_widths_predictions = factor_params[:, :, 3]
        centers_predictions = centers_predictions[:, :, :, 0]
        log_widths_predictions = log_widths_predictions[:, :, 0]

        mean_factors = tfa_models.radial_basis(dtfa.voxel_locations,
                                               centers_predictions.data,
                                               log_widths_predictions.data)[0, :, :]

        mean_brain = mean_weight @ mean_factors

        image = utils.cmu2nii(mean_brain.data.numpy(),
                              dtfa.voxel_locations.numpy(),
                              dtfa._templates[0])
        filename = 'results/subject_' + str(subjects[p]) + '_task_' + str(tasks[s]) + '_generated_interaction_brain.png'
        plot = niplot.plot_glass_brain(
            image, title='subject_' + str(subjects[p]) + '_task_' + str(tasks[s]), plot_abs=False,
            colorbar=True, symmetric_cbar=True, output_file=filename
        )

for p in range(len(subjects)):
    use_subject = subjects[p]
    print(use_subject)
    # for stimulus in ['spider','social','heights']:
    for (t, stimulus) in enumerate(task_category):
        # temp_z_ps = z_i_mu[4,:]
        idx = [id for (id, inter) in enumerate(interactions) if inter == (use_subject, stimulus + '_high')][0]
        print(idx)
        temp_z_ps_high = z_i_mu[idx]
        idx = [id for (id, inter) in enumerate(interactions) if inter == (use_subject, stimulus + '_low')][0]
        temp_z_ps_low = z_i_mu[idx]
        use_points = list(getEquidistantPoints(temp_z_ps_low.data, temp_z_ps_high.data, 10))
        temp_pf_mu = z_pf_mu[p, :]
        temp_pw_mu = z_pw_mu[p, :]
        temp_s_mu = z_s_mu[t, :]
        filenames = []
        for (i, points) in enumerate(use_points):
            joint_embed = torch.cat((temp_pw_mu, temp_s_mu, torch.tensor(points)), dim=-1)
            weight_predictions = dtfa.decoder.weights_embedding(joint_embed).view(
                -1, dtfa.num_factors, 2
            )
            mean_weight = weight_predictions[:, :, 0]

            factor_params = dtfa.decoder.factors_embedding(temp_pf_mu).view(
                -1, dtfa.num_factors, 4, 2
            )
            centers_predictions = factor_params[:, :, :3, 0]
            log_widths_predictions = factor_params[:, :, 3, 0]
            mean_factors = tfa_models.radial_basis(dtfa.voxel_locations,
                                                   centers_predictions.data,
                                                   log_widths_predictions.data)[0, :, :]
            mean_brain = mean_weight @ mean_factors
            vmax = torch.max(mean_brain).data.numpy()
            # if vmax < 1:
            #     vmax = 1
            image = utils.cmu2nii(mean_brain.data.numpy(),
                                  dtfa.voxel_locations.numpy(),
                                  dtfa._templates[0])
            filenames.append(
                'results/gifs/subject_' + str(use_subject) + '_stimulus_' + str(stimulus) + '_' + str(i) + '.png')
            plot = niplot.plot_glass_brain(
                image, plot_abs=False, colorbar=True, symmetric_cbar=True,
                title="Mean Image of Interaction %d" % i,
                vmin=-3, vmax=3, output_file=filenames[-1])

        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave('results/gifs/subject_' + str(use_subject) + '_stimulus_' + stimulus + '.gif', images)
r = 3