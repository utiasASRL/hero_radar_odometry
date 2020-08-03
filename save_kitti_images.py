import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import glob
import itertools
import pykitti
import torch
import argparse
import json
from utils.helper_func import *

parser = argparse.ArgumentParser(description='Point Cloud Registration')
parser.add_argument('--config', default=None, type=str,
                    help='config file path (default: None)')
parser.add_argument('--save_dir', type=str, default='.', metavar='N',
                    help='Save to current directory by default')
parser.add_argument('--sequence_number', type=str, default='00', metavar='N',
                    help="Sequence number. String, e.g '00'")
parser.add_argument('--num_frames', type=int, default=10, metavar='N',
                    help='Number of frames to write. Integer value: e.g 10. Set to -1 for all frames.')

args = parser.parse_args()

with open(args.config) as f:
    config = json.load(f)

save_dir = args.save_dir
dataset = pykitti.odometry(config["dataset"]["data_dir"], args.sequence_number)

# take point cloud
num_poses = len(dataset.poses)
sample_n = args.num_frames
if sample_n == -1:
    sample_n = num_poses
assert sample_n <= num_poses, "Sequence does NOT have enough frames!"
sample_idx = np.arange(sample_n)

# load image configs
azi_res = config["dataset"]["images"]["azi_res"]
azi_min = config["dataset"]["images"]["azi_min"]
azi_max = config["dataset"]["images"]["azi_max"]
ele_res = config["dataset"]["images"]["ele_res"]
ele_min = config["dataset"]["images"]["ele_min"]
ele_max = config["dataset"]["images"]["ele_max"]
input_channel = config["dataset"]["images"]["input_channel"]

# image sizes
horizontal_pix = np.int32((azi_max - azi_min) / azi_res)
vertical_pix = np.int32((ele_max - ele_min) / ele_res)
geometry_img = np.zeros((3, vertical_pix, horizontal_pix), dtype=np.float32)

# inspect vertex map
for sample_i in sample_idx:
    sample_velo = dataset.get_velo(sample_i)

    if config["dataset"]["images"]["map_laser_to_row"]:
        geometry_img, input_img = pc2img_laser_to_row(sample_velo, azi_res, azi_min, azi_max,
                                                      input_channel, horizontal_pix)
    else:
        geometry_img, input_img = pc2img(sample_velo, geometry_img, azi_res, azi_min, azi_max,
                                        ele_res, ele_min, ele_max, input_channel,
                                        horizontal_pix, vertical_pix)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if "intensity" in input_channel:
        plt.imsave('{}/intensity_{}.png'.format(save_dir, sample_i), input_img[3])
    if "range" in input_channel:
        plt.imsave('{}/range_{}.png'.format(save_dir, sample_i), input_img[4])

print("====================")
print("===WRITE COMPLETE===")
print("====================")