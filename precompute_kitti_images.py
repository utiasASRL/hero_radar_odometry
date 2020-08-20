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
parser.add_argument('--data_dir', default=None, type=str,
                    help='config file path (default: None)')
parser.add_argument('--save_dir', type=str, default='.', metavar='N',
                    help='Save to current directory by default')
# parser.add_argument('--sequence_number', type=str, default='00', metavar='N',
#                     help="Sequence number. String, e.g '00'")
# parser.add_argument('--num_frames', type=int, default=10, metavar='N',
#                     help='Number of frames to write. Integer value: e.g 10. Set to -1 for all frames.')

args = parser.parse_args()

# set image configs
azi_res = 0.5
azi_min = -180.0
azi_max = 180.0
ele_res = 0.5   # ele parameters don't matter if mapping laser to row
ele_min = -25.0
ele_max = 3.0
input_channel = ['vertex', 'intensity', 'range'] # fix input to all channels

# image sizes
horizontal_pix = np.int32((azi_max - azi_min) / azi_res)
vertical_pix = np.int32((ele_max - ele_min) / ele_res)
geometry_img = np.zeros((3, vertical_pix, horizontal_pix), dtype=np.float32)

# set sequences
seq_list = ["00","01","02","03","04","05","06","07","08","09","10"]
for seq_num in seq_list:
    # file directories
    save_dir = os.path.join(args.save_dir, seq_num)
    dataset = pykitti.odometry(args.data_dir, seq_num)

    # take point cloud
    num_poses = len(dataset.poses)
    sample_n = num_poses
    sample_idx = np.arange(sample_n)
    for sample_i in sample_idx:
        sample_velo = dataset.get_velo(sample_i)

        # if config["dataset"]["images"]["map_laser_to_row"]:
        geometry_img, input_img = pc2img_laser_to_row(sample_velo, azi_res, azi_min, azi_max,
                                                      input_channel, horizontal_pix)
        # else:
        #     geometry_img, input_img = pc2img(sample_velo, geometry_img, azi_res, azi_min, azi_max,
        #                                     ele_res, ele_min, ele_max, input_channel,
        #                                     horizontal_pix, vertical_pix)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # plt.imsave('{}/intensity_{}.png'.format(save_dir, sample_i), input_img)
        np.save('{}/{}.npy'.format(save_dir, sample_i), input_img)
        print(sample_i, sample_n)


print("====================")
print("===WRITE COMPLETE===")
print("====================")