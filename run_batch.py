import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

from datasets.oxford import get_dataloaders
from datasets.boreas import get_dataloaders_boreas
from datasets.radar import radar_polar_to_cartesian
from networks.batch import Batch
from utils.utils import computeMedianError, computeKittiMetrics, saveKittiErrors, save_in_yeti_format, get_T_ba
from utils.utils import load_icra21_results, getStats, get_inverse_tf, get_folder_from_file_path
from utils.vis import convert_plt_to_img, plot_sequences


def convert_lidar_line_to_pose(line):
    line = line.replace('\n', '').split(' ')
    line = [float(i) for i in line]
    T = np.eye(4)
    k = 0
    for i in range(3):
        for j in range(4):
            T[i, j] = line[k]
            k += 1
    return T

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/boreas_batch_local.json', type=str, help='config file path')
    parser.add_argument('--pretrain', default=None, type=str, help='pretrain checkpoint path')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    # model
    model = Batch(config)

    if config['dataset'] == 'oxford':
        _, _, test_loader = get_dataloaders(config)
    elif config['dataset'] == 'boreas':
        _, _, test_loader = get_dataloaders_boreas(config)

    # load lidar / vehicle (robot) calibration
    T_applanix_lidar = np.loadtxt('comparison/T_applanix_lidar.txt', dtype=np.float32)
    yfwd2xfwd = np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    T_robot_lidar = yfwd2xfwd @ T_applanix_lidar
    T_lidar_robot = np.linalg.inv(T_robot_lidar)
    model.solver.setExtrinsicLidarVehicle(T_lidar_robot)

    seq_name = test_loader.dataset.sequences[0]

    model.solver.setRadar2DFlag(True)
    for batchi, batch in enumerate(test_loader):
        print('{} / {}'.format(batchi, len(test_loader)))
        # if batchi < 50:
        #     continue
        # if batchi > 560:
        # if batchi > 345:
        if batchi > 150:
            break

        if batchi == 0:
            save_for_plotting = True
        else:
            save_for_plotting = False

        # load next frame pair
        # if batchi == 345:
        # // incomplete radar scan at 345
        model.add_frame_pair(batch, save_for_plotting)

    # model.plot_frame(0)
    # load and apply lidar poses
    file = "comparison/odometry_result-boreas-2021-07-12-15-05.txt"
    with open(file, 'r') as f:
        lines = f.readlines()
    lidar_poses = [convert_lidar_line_to_pose(line) for line in lines]
    with open('comparison/timestamps.txt', 'r') as f:
        lines = f.readlines()
    lidar_times = [int(_) + model.time_offset for _ in lines]

    # add lidar pose measurements
    model.solver.addLidarPoses(lidar_poses, lidar_times)
    # model.solver.addLidarPosesRel(lidar_poses, lidar_times)

    # # first solve with just radar odometry (lock the extrinsic to initial guess)
    # model.solver.setFirstPoseLock(True)
    # model.solver.setLockExtrinsicState(True)
    # model.solver.setRadarFlag(True)
    # model.solver.setLidarFlag(False)
    # model.solver.optimize()
    #
    # # get path
    # radar_path = np.zeros((model.solver.getTrajLength(), 3), dtype=np.float32)
    # radar_times = np.zeros((model.solver.getTrajLength()), dtype=np.float32)
    # model.solver.getPath(radar_path, radar_times)
    #
    # # solve with lidar poses
    # model.solver.setFirstPoseLock(True)
    # model.solver.setLockExtrinsicState(True)
    # model.solver.setRadarFlag(False)
    # model.solver.setLidarFlag(True)
    # model.solver.optimize()
    #
    # lidar_path = np.zeros((model.solver.getTrajLength(), 3), dtype=np.float32)
    # lidar_times = np.zeros((model.solver.getTrajLength()), dtype=np.float32)
    # model.solver.getPath(lidar_path, lidar_times)

    # solve for extrinsic
    model.solver.setFirstPoseLock(False)
    model.solver.setLockExtrinsicState(False)
    model.solver.setRadarFlag(True)
    model.solver.setLidarFlag(True)
    model.solver.optimize()

    path = np.zeros((model.solver.getTrajLength(), 3), dtype=np.float32)
    times = np.zeros((model.solver.getTrajLength()), dtype=np.float32)
    model.solver.getPath(path, times)

    # plot
    plt.figure()
    ax = plt.axes()
    plt.axis('equal')
    # plt.plot(radar_path[:, 0], radar_path[:, 1], 'b.', label='radar path')
    # plt.plot(lidar_path[:, 0], lidar_path[:, 1], 'k.', label='lidar path')
    plt.plot(path[:, 0], path[:, 1], 'k.', label='path')
    # ax.legend()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    plt.show()

