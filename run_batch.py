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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/boreas_local.json', type=str, help='config file path')
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

    seq_name = test_loader.dataset.sequences[0]

    T_gt = []
    T_pred = []

    for batchi, batch in enumerate(test_loader):
        print('{} / {}'.format(batchi, len(test_loader)))
        # if batchi < 344:
        #     continue
        if batchi > 560:
            break

        # load next frame pair
        # if batchi == 345:
        # // incomplete radar scan at 345
        model.add_frame_pair(batch)

    # run batch solve
    model.solver.optimize()

    # get path
    path = np.zeros((model.solver.getTrajLength(), 3), dtype=np.float32)
    model.solver.getPath(path)

    # plot
    plt.figure()
    ax = plt.axes()
    plt.axis('equal')
    plt.plot(path[:, 0], path[:, 1], 'k.', label='estimated path')
    # ax.legend()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    plt.imshow()

