import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

def visualize_data_distribution(ids, poses_log, results_dir, dataset_name):
    trans_norms = [torch.norm(v[0][0:3]).item() for k, v in poses_log.items() if k in ids]
    rot_norms = [torch.norm(v[0][3:6]).item() for k, v in poses_log.items() if k in ids]

    xedges = np.arange(0.0, 1.0, 0.01)
    yedges = np.arange(0.0, 0.1, 0.005)

    H, xedges, yedges = np.histogram2d(np.asarray(trans_norms), np.asarray(rot_norms), bins=(xedges, yedges))
    H = H.T  # Let each row list bins with common y range.

    plt.figure(figsize=(20, 10))
    im = plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.xlabel('Norm, translation (meters)')
    plt.ylabel('Norm rotation {rad}')
    plt.title('Distribution of data samples translation/rotation norm - {}'.format(dataset_name))
    plt.colorbar(im);

    plt.savefig(results_dir + dataset_name + '_distribution.png', format='png')

def rms_errors(outputs, targets):
    return torch.sqrt(torch.mean((targets - outputs)**2, dim=0))

def prepare_device(device_ids_use):
    """
    Setup GPU device if available,
    """
    num_gpu = torch.cuda.device_count()
    num_gpu_use = len(device_ids_use)

    print("num_gpu: {}".format(num_gpu))
    print("num_gpu_use: {}".format(num_gpu_use))

    if num_gpu_use > 0 and num_gpu == 0:
        print("Warning: There\'s no GPU available on this machine.")
        num_gpu_use = 0

    if num_gpu_use > num_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but "
              "only {} are available.".format(num_gpu_use, num_gpu))
        num_gpu_use = num_gpu

    device = torch.device('cuda:{}'.format(0) if num_gpu_use > 0 else 'cpu')

    print("device: {}".format(device))
    print("device_ids_use: {}".format(device_ids_use))

    return device, device_ids_use

def normalize_coords(coords_2D, batch_size, height, width):
    # B x N x 2
    u_norm = (2 * coords_2D[:, :, 0].reshape(batch_size, -1) / (width - 1)) - 1
    v_norm = (2 * coords_2D[:, :, 1].reshape(batch_size, -1) / (height - 1)) - 1

    # WARNING: grid_sample expects the normalized coordinates as (u, v)
    return torch.stack([u_norm, v_norm], dim=2)  # B x N x 2

def sample_desc(feature_map, keypoints):

    batch_size, channels, height, width = feature_map.size()

    # Get descriptors only for keypoints
    # Normalize 2D coordinates for sampling [-1, 1]
    keypoints_norm = normalize_coords(keypoints, batch_size, height, width).unsqueeze(1) # B x 1 x N x 2

    # Sample descriptors for the chosen test keypoints in the src image
    descriptors = F.grid_sample(feature_map, keypoints_norm, mode='bilinear')  # B x C x 1 x N
    descriptors = descriptors.reshape(batch_size, channels, keypoints.size(1)) # B x C x N

    return descriptors

def get_norm_desc(feature_map, sample=False, keypoints=None):

    batch_size, channels, height, width = feature_map.size()

    if sample:
        # Get descriptors only for keypoints
        # Normalize 2D coordinates for sampling [-1, 1]
        keypoints_norm = normalize_coords(keypoints, batch_size, height, width).unsqueeze(1) # B x 1 x N x 2

        # Sample descriptors for the chosen test keypoints in the src image
        descriptors = F.grid_sample(feature_map, keypoints_norm, mode='bilinear')  # B x C x 1 x N
        descriptors = descriptors.reshape(batch_size, channels, keypoints.size(1)) # B x C x N
    else:
        # Get descriptors for the whole image
        descriptors = feature_map.reshape(batch_size, channels, height * width)    # B x C x HW

    descriptors_mean = torch.mean(descriptors, dim=1, keepdim=True)           # B x 1 x (N or HW)
    descriptors_std = torch.std(descriptors, dim=1, keepdim=True)             # B x 1 x (N or HW)
    descriptors_norm = (descriptors - descriptors_mean) / descriptors_std     # B x C x (N or HW)

    return descriptors_norm

def get_scores(score_map, keypoints):

    batch_size, _, height, width = score_map.size()
    n_points = keypoints.size(1)

    # Normalize 2D coordinates for sampling [-1, 1]
    keypoints_norm = normalize_coords(keypoints, batch_size, height, width).unsqueeze(1) # B x 1 x N x 2
    scores = F.grid_sample(score_map, keypoints_norm, mode='bilinear')                        # B x 1 x 1 x N
    scores = scores.reshape(batch_size, 1, n_points)

    return scores

def l2_norm(descriptors_src, descriptors_trg):

    batch_size, channels, n_points = descriptors_trg.size()

    # Repeat descriptor vectors along different dimensions to get matrices
    desc_trg_matrix = descriptors_trg.unsqueeze(2).expand(batch_size, channels, n_points, n_points) # B x C x N x N
    desc_src_matrix = descriptors_src.unsqueeze(3).expand(batch_size, channels, n_points, n_points) # B x C x N x N

    # Get the errors between all combinations of source and target descriptors.
    desc_diff = desc_trg_matrix - desc_src_matrix

    # Reorder the dimensons and reshape the tensor so we can take the dot product for each of the errors individually to get e^T * e
    desc_diff = desc_diff.transpose(2,1).contiguous().transpose(3,2).contiguous().reshape(batch_size * n_points * n_points, channels) # BNN x C

    # e^T * e with dimensions BNN x 1 x C * BNN x C x 1
    l2 = torch.matmul(desc_diff.unsqueeze(1), desc_diff.unsqueeze(2)) / desc_diff.size(1) # BNN x 1
    l2 = l2.reshape(batch_size, n_points, n_points)

    return l2

def zn_desc(desc):
    '''
    Zero-normalize the descriptor along dim-1. Assume desc BxCxN
    @param desc: un-normalized desc
    @return: zero-normalized desc
    '''
    desc = (desc - torch.mean(desc, dim=1, keepdim=True)) / torch.std(desc, dim=1, keepdim=True)
    return desc

def T_inv(transformation_mat):
    batch_size = transformation_mat.size(0)
    identity = torch.eye(4)
    identity = identity.reshape((1, 4, 4))
    T_inv = identity.repeat(batch_size, 1, 1).cuda()
    C = transformation_mat[:,:3,:3]
    t = transformation_mat[:,:3,3:]
    T_inv[:,:3,:3] = C.transpose(2,1).contiguous()
    T_inv[:,:3,3:] = -C.transpose(2,1).contiguous() @ t
    return T_inv
