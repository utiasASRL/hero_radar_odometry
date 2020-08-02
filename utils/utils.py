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

# from src.dataset import Dataset

class MELData():

    def __init__(self):
        self.mean = [0.0, 0.0, 0.0]
        self.std_dev = [0.0, 0.0, 0.0]
        self.precision = 0.0

        self.train_labels_se3 = {}
        self.train_labels_log = {}
        self.train_ids = []

        self.valid_labels_se3 = {}
        self.valid_labels_log = {}
        self.valid_ids = []

        self.test_labels_se3 = {}
        self.test_labels_log = {}
        self.test_ids = []

        self.paths = []
        self.path_loc_labels_se3 = {}
        self.path_loc_labels_log = {}
        self.path_loc_ids = {}

        self.path_vo_labels_se3 = {}
        self.path_vo_labels_log = {}
        self.path_vo_ids = {}

# def compute_mean_std(mel_data, data_dir, use_stereo, device, num_pairs):
#     start = time.time()
#
#     image_params = {'num_dof': 6,
#                     'data_dir': data_dir,
#                     'desired_height': 384,
#                     'desired_width': 512,
#                     'use_stereo': use_stereo,
#                     'use_ref_images': False,
#                     'use_normalization': False,
#                     'use_crop': False,
#                     'use_disparity': False}
#
#     num_channels = 6 * num_pairs if image_params['use_stereo'] else 3 * num_pairs
#
#     params = {'batch_size': 128,
#               'shuffle': False,
#               'num_workers': 12}
#
#     training_set = Dataset(num_pairs, **image_params)
#     training_set.load_mel_data(mel_data, 'train')
#     training_generator = data.DataLoader(training_set, **params)
#
#     fst_moment = torch.empty(num_channels)
#     snd_moment = torch.empty(num_channels)
#
#     fst_moment = fst_moment.to(device)
#     snd_moment = snd_moment.to(device)
#
#     i = 0
#     cnt = 0
#     for images, _, _, _, _ in training_generator:
#         i += 1
#
#         images = images[:, 0:num_channels, :, :].to(device)
#
#         b, c, h, w = images.size()
#         nb_pixels = b * h * w
#         sum_ = torch.sum(images, dim=[0,2,3])
#         sum_of_square = torch.sum(images ** 2, dim=[0,2,3])
#         fst_moment = ((cnt * fst_moment) + sum_) / (cnt + nb_pixels)
#         snd_moment = ((cnt * snd_moment) + sum_of_square) / (cnt + nb_pixels)
#
#         cnt += nb_pixels
#
#     pop_mean = fst_moment
#     pop_std = torch.sqrt(snd_moment - (fst_moment ** 2))
#
#     pop_mean_final = [0, 0, 0]
#     pop_std_final = [0, 0, 0]
#
#     num_comp = int(num_channels / 3)
#
#     for i in range(3):
#         for j in range(num_comp):
#             pop_mean_final[i] += pop_mean[(j * 3) + i].item()
#             pop_std_final[i] += pop_std[(j * 3) + i].item()
#
#         pop_mean_final[i] = pop_mean_final[i] / num_comp
#         pop_std_final[i] = pop_std_final[i] / num_comp
#
#     print('Training images mean: ' + str(pop_mean_final))
#     print('Training images std: ' + str(pop_std_final))
#     print('Computing mean/std took: ' + str(time.time() - start))
#
#     return pop_mean_final, pop_std_final
#
# def compute_prec_test(mel_data, data_dir, device=None, num_pairs=3):
#     start = time.time()
#
#     image_params = {'num_dof': 6,
#                     'data_dir': data_dir,
#                     'desired_height': 384,
#                     'desired_width': 512,
#                     'use_stereo': False,
#                     'use_ref_images': False,
#                     'use_normalization': False,
#                     'use_crop': False,
#                     'use_disparity': False}
#
#     params = {'batch_size': 128,
#               'shuffle': False,
#               'num_workers': 12}
#
#     training_set = Dataset(num_pairs, **image_params)
#     training_set.load_mel_data(mel_data, 'train')
#     training_generator = data.DataLoader(training_set, **params)
#
#     targets = torch.FloatTensor(len(training_set), 6).zero_()
#     targets2 = torch.FloatTensor(len(training_set), 6).zero_()
#
#     ind = 0
#     for _, _, _, log_poses_all in training_generator:
#         log_poses = log_poses_all[0]
#         batch_size = log_poses.size(0)
#
#         for i in range(batch_size):
#             trans_norm = torch.norm(log_poses[i, 0:3])
#             rot_norm = torch.norm(log_poses[i, 3:6])
#             all_norm = torch.norm(log_poses[i, :])
#
#             if trans_norm == 0.0:
#                 trans_norm = 1.0
#
#             if rot_norm == 0.0:
#                 rot_norm = 1.0
#
#             if all_norm == 0.0:
#                 all_norm = 1.0
#
#             targets[ind + i, :] = torch.div(log_poses[i, :], torch.tensor(([trans_norm]*3) + ([rot_norm]*3)))
#             targets2[ind + i, :] = log_poses[i, :].float() * (1.0 / all_norm)
#
#
#         ind += batch_size
#
#     precision = torch.from_numpy(np.linalg.inv(np.cov(targets.numpy().T))).float()
#     precision2 = torch.from_numpy(np.linalg.inv(np.cov(targets2.numpy().T))).float()
#     print('Precision: ' + str(precision2))
#     print('Took: ' + str(time.time() - start))
#
#     return precision, precision2

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

# def visualize_data_histogram(mel_data, data_dir, results_dir, dataset_name, device, num_pairs=3 ,path_name='', index=0):
#     name = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
#     unit = ['meters', 'meters', 'meters', 'radians', 'radians', 'radians']
#
#     image_params = {'num_dof': 6,
#                     'data_dir': data_dir,
#                     'desired_height': 384,
#                     'desired_width': 512,
#                     'use_stereo': False,
#                     'use_ref_images': False,
#                     'use_normalization': False,
#                     'use_crop': False,
#                     'use_disparity': False}
#
#     params = {'batch_size': 128,
#               'shuffle': False,
#               'num_workers': 6}
#
#     training_set = Dataset(num_pairs, **image_params)
#     training_set.load_mel_data(mel_data, dataset_name, path_name, index)
#     training_generator = data.DataLoader(training_set, **params)
#
#     ind = 0
#     num_poses = len(training_set)
#     poses = torch.zeros((num_poses, 6), device=device)
#     for _, _, _, log_poses_all in training_generator:
#         log_poses = log_poses_all[0].to(device)
#         batch_size = log_poses.size(0)
#         poses[ind:ind + batch_size, :] = torch.abs(log_poses)
#         ind += batch_size
#
#     edges = [np.arange(0.0, 0.15, 0.006), \
#              np.arange(0.0, 0.15, 0.006), \
#              np.arange(0.0, 0.5, 0.02), \
#              np.arange(0.0, 0.05, 0.002), \
#              np.arange(0.0, 0.05, 0.002), \
#              np.arange(0.0, 0.05, 0.002)]
#
#     for i in range(6):
#         plt.figure()
#         plt.hist(poses[:, i].detach().cpu().numpy(), bins=(edges[i]))
#         plt.xlabel('{}'.format(unit[i]))
#         plt.ylabel('count')
#         plt.title('Data distribution - {}'.format(name[i]))
#         if dataset_name == 'test_paths':
#             plt.savefig('{0}{1}_data_hist_{2}_{3}.png'.format(results_dir, dataset_name, name[i], index), format='png')
#         else:
#             plt.savefig('{0}{1}_data_hist_{2}.png'.format(results_dir, dataset_name, name[i]), format='png')
#         plt.close()

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

def get_norm_descriptors(feature_map, sample=False, keypoints=None):

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