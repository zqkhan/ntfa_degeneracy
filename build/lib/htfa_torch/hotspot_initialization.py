from __future__ import division
import numpy as np
import scipy.optimize as optimize

def initialize_centers_widths_weights(activations,locations,num_factors):
    temp_data  = []
    temp_R = []
    for i in range(len(activations)):
        temp_data.append(activations[i].numpy())
        temp_R.append(locations[i].numpy())
    activations = temp_data
    locations = temp_R
    del temp_data,temp_R
    mean_activations,mean_locations = mean_image(activations,locations)
    template_center_mean,template_width_mean = hotspot_initialization(mean_activations,
                                                                      mean_locations,
                                                                      num_factors)
    F = radial_basis(mean_locations,template_center_mean,template_width_mean).T
    trans_F = F.T.copy()
    template_weights_mean = np.linalg.solve(trans_F.dot(F), trans_F.dot(mean_activations))
    subject_weights_mean = []
    subject_center_mean = []
    subject_width_mean = []
    for i in range(len(activations)):
        F = radial_basis(locations[i], template_center_mean, template_width_mean).T
        trans_F = F.T.copy()
        subject_weights_mean.append(np.linalg.solve(trans_F.dot(F), trans_F.dot(mean_activations)))
        subject_center_mean.append(template_center_mean)
        subject_width_mean.append(template_width_mean)

    return template_center_mean,template_width_mean,template_weights_mean,\
           np.array(subject_center_mean),np.array(subject_width_mean),np.array(subject_weights_mean)


def hotspot_initialization(activations, locations , num_factors):
    mean_activations = abs(activations - np.nanmean(activations))
    centers = np.zeros(shape=(num_factors,locations.shape[1]))
    widths = np.zeros(shape=(num_factors,))

    for k in range(num_factors):
        ind = np.nanargmax(mean_activations)
        centers[k,:] = locations[ind,:]
        widths[k] = init_width(activations,locations,activations[ind],centers[k,:])
        mean_activations = mean_activations - radial_basis(locations,centers[k,:],widths[k])
    return centers,widths

def mean_image(activations, locations):

    mean_locations = locations[0]

    for i in range(1, len(locations)):
        mean_locations = np.vstack([mean_locations, locations[i]])   ## fix this to accomodate differing lengths

    mean_locations = np.unique(mean_locations,axis=0)
    mean_activations = np.zeros(shape=(mean_locations.shape[0],))
    n = np.zeros(shape=(mean_activations.shape))

    for i in range(len(activations)):
        C = intersect(mean_locations,locations[i])
        mean_locations_ind = get_common_indices(mean_locations,C)
        subject_locations_ind = get_common_indices(locations[i],C)
        mean_locations_ind = np.sort(mean_locations_ind)
        subject_locations_ind = np.sort(subject_locations_ind)
        mean_activations[mean_locations_ind] = mean_activations[mean_locations_ind] + \
                                               np.mean(activations[i][:,subject_locations_ind],axis=0)
        n[mean_locations_ind] = n[mean_locations_ind] + 1
    mean_activations = mean_activations/n

    return mean_activations,mean_locations


def init_width(activations,locations,weight,c):

    start_width = 0
    objective = lambda w: np.sum(np.abs(activations - weight*radial_basis(locations,c,w)))
    result = optimize.minimize(objective,x0=start_width)

    return result.x

def radial_basis(locations, centers, log_widths):
    """The radial basis function used as the shape for the factors"""
    # V x 3 -> 1 x V x 3
    locations = np.expand_dims(locations,0)
    if len(centers.shape) > 3:
        # 1 x V x 3 -> 1 x 1 x V x 3
        locations = np.expand_dims(locations,0)
    # S x K x 3 -> S x K x 1 x 3
    centers = np.expand_dims(centers,len(centers.shape)-1)
    # S x K x V x 3
    delta2s = (locations - centers)**2
    # S x K  -> S x K x 1
    log_widths = np.expand_dims(log_widths,len(log_widths.shape))
    return np.exp(-delta2s.sum(len(delta2s.shape) - 1) / np.exp(log_widths))

def intersect(A,B):
    return np.array([x for x in set(tuple(x) for x in A) & set(tuple(x) for x in B)])

def get_common_indices(X,C):
    return np.where((X==C[:,None]).all(-1))[1]
