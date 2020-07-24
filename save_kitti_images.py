import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import h5py
import glob
import itertools
import pykitti
import torch
import argparse

parser = argparse.ArgumentParser(description='Point Cloud Registration')
parser.add_argument('--save_dir', type=str, default='.', metavar='N',
                    help='Save to current directory by default')
parser.add_argument('--sequence_number', type=str, default='00', metavar='N',
                    help="Sequence number. String, e.g '00'")
parser.add_argument('--num_frames', type=int, default=10, metavar='N',
                    help='Number of frames to write. Integer value: e.g 10')

args = parser.parse_args()

base_dir = '/mnt/ssd1/research/dataset/KITTI/dataset'

save_dir = args.save_dir

dataset = pykitti.odometry(base_dir, args.sequence_number)

# take point cloud
num_poses = len(dataset.poses)
sample_n = args.num_frames
assert sample_n <= num_poses, "Sequence does NOT have enough frames!"
sample_idx = np.arange(sample_n)

# image to store 2D data
azi_res = 0.5
azi_min = -180.0
azi_max = 180.0
ele_res = 0.5
ele_min = -25.0
ele_max = 3.0
horizontal_pix = np.int32((azi_max - azi_min) / azi_res)
vertical_pix = np.int32((ele_max - ele_min) / ele_res)
num_channel = 3
vertex_img = np.zeros((vertical_pix, horizontal_pix, num_channel))
range_img = np.zeros((vertical_pix, horizontal_pix))
intensity_img = np.zeros((vertical_pix, horizontal_pix))

# inspect vertex map
for sample_i in sample_idx:
    sample_velo = dataset.get_velo(sample_i)

    sample_velo_xyz = sample_velo[:,:3]
    sample_velo_i = sample_velo[:,3]

    # sort the points based on range
    r = np.sqrt(np.sum(sample_velo_xyz ** 2, axis=1))
    order = np.argsort(r)
    sample_velo = sample_velo[order[::-1]]
    sample_velo_xyz = sample_velo_xyz[order[::-1]]
    sample_velo_i = sample_velo_i[order[::-1]]
    r = r[order[::-1]]

    # compute azimuth and elevation
    azimuth = np.rad2deg(np.arctan2(sample_velo_xyz[:,1], sample_velo_xyz[:,0]))

    xy = np.sqrt(sample_velo_xyz[:,0] ** 2 + sample_velo_xyz[:,1] ** 2)
    elevation = np.rad2deg(np.arctan2(sample_velo_xyz[:,2], xy))

    # reject points outside the field of view
    ids = (azimuth >= azi_min) * (azimuth <= azi_max) * (elevation >= ele_min) * (elevation <= ele_max)

    # compute u,v
    u = np.int32((0.5 * (azi_max - azi_min) - azimuth[ids]) // azi_res)
    u[u == horizontal_pix] = 0
    v = np.int32((ele_max - elevation[ids]) // ele_res)

    # write range image
    range_img[v, u] = r[ids]

    # write intensity image
    intensity_img[v, u] = sample_velo_i[ids]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.imsave('{}/range_{}.png'.format(save_dir, sample_i), range_img)
    plt.imsave('{}/intensity_{}.png'.format(save_dir, sample_i), intensity_img)

print("====================")
print("===WRITE COMPLETE===")
print("====================")