import signal
import os
from os import makedirs, remove
from os.path import exists, join
import time
import argparse
import json
import matplotlib.pyplot as plt
from evtk.hl import pointsToVTK, linesToVTK

# Dataset
from datasets.custom_sampler import *
from datasets.kitti import *
from utils.config import *
from torch.utils.data import DataLoader
import torch.nn.functional as F

# network
from networks.f2f_pose_model import F2FPoseModel

# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='results/ir_super_w6_mah_00/config.json', type=str,
                      help='config file path (default: config/steam_f2f.json)')

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    config['dataset']['data_dir'] = '/home/david/Data/kitti'

    # Initialize datasets
    train_dataset = KittiDataset(config, set='test')
    train_sampler = WindowBatchSampler(batch_size=1,
                                       window_size=2,
                                       seq_len=train_dataset.seq_len,
                                       drop_last=True)


    # Initialize the dataloader
    training_loader = DataLoader(train_dataset,
                                 batch_sampler=train_sampler,
                                 num_workers=0,
                                 pin_memory=True)

    # gpu
    device = torch.device("cuda:0")

    # load checkpoint
    previous_training_path = config['previous_session']
    chosen_chkp = 'chkp.tar'
    chosen_chkp = os.path.join('results', previous_training_path, chosen_chkp)
    checkpoint = torch.load(chosen_chkp)

    # set output path
    output_path = os.path.join('plot', previous_training_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # network
    net = F2FPoseModel(config,
                         config['test_loader']['window_size'],
                         config['test_loader']['batch_size'])
    net.to(device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()


    # load random pair
    for i_batch, data in enumerate(training_loader):

        fid = int(data['f_ind'][0])

        # print keypoints
        src_coords, tgt_coords, weights = net.forward_keypoints(data)

        # get src points
        points1 = src_coords[0, :, :].transpose(0, 1)

        # check for no returns in src keypoints
        nr_ids = torch.nonzero(torch.sum(points1, dim=1), as_tuple=False).squeeze()
        points1 = points1[nr_ids, :]

        # get tgt points
        points2 = tgt_coords[0, :, nr_ids].transpose(0, 1)

        # match consistency
        w12 = net.softmax_matcher_block.match_vals[0, nr_ids, :] # N x M
        # w12 = torch.zeros((5, 4))
        _, ind2to1 = torch.max(w12, dim=1)  # N
        _, ind1to2 = torch.max(w12, dim=0)  # M
        mask = torch.eq(ind1to2[ind2to1], torch.arange(ind2to1.__len__(), device=ind1to2.device))
        mask_ind = torch.nonzero(mask, as_tuple=False).squeeze()
        points1 = points1[mask_ind, :]
        points2 = points2[mask_ind, :]

        T_iv = data['T_iv'].cuda()
        T_21 = net.se3_inv(T_iv[1, :, :])@T_iv[0, :, :]
        points1_in_2 = points1@T_21[:3, :3].T + T_21[:3, 3].unsqueeze(0)

        # convert to numpy
        points1_in_2 = points1_in_2.detach().cpu().numpy()
        points2 = points2.detach().cpu().numpy()

        # print points
        x = np.asarray(points1_in_2[:, 0], order='C')
        y = np.asarray(points1_in_2[:, 1], order='C')
        z = np.asarray(points1_in_2[:, 2], order='C')
        pointsToVTK("{}/points1_{}".format(output_path, fid), x, y, z)
        x = np.asarray(points2[:, 0], order='C')
        y = np.asarray(points2[:, 1], order='C')
        z = np.asarray(points2[:, 2], order='C') + 50
        pointsToVTK("{}/points2_{}".format(output_path, fid), x, y, z)

        # association lines
        skip = 10
        points1_in_2 = points1_in_2[::skip]
        points2 = points2[::skip]
        N = points1_in_2.shape[0]
        x = np.zeros((2*N,), order='C')
        y = np.zeros((2*N,), order='C')
        z = np.zeros((2*N,), order='C')
        x[::2] = points1_in_2[:, 0]
        y[::2] = points1_in_2[:, 1]
        z[::2] = points1_in_2[:, 2]
        x[1::2] = points2[:, 0]
        y[1::2] = points2[:, 1]
        z[1::2] = points2[:, 2] + 50

        linesToVTK("{}/img_{}".format(output_path, fid), x, y, z)

        # print raw image pc
        geometry_img = data['geometry'][0, :, :].cuda()
        image_pc = torch.reshape(geometry_img, (geometry_img.shape[0], -1)).transpose(0, 1)
        image_pc = image_pc@T_21[:3, :3].T + T_21[:3, 3].unsqueeze(0)
        image_pc = image_pc.detach().cpu().numpy()
        x = np.asarray(image_pc[:, 0], order='C')
        y = np.asarray(image_pc[:, 1], order='C')
        z = np.asarray(image_pc[:, 2], order='C')
        pointsToVTK("{}/raw1_{}".format(output_path, fid), x, y, z)

        # plt.imsave('{}/detect_weight{}.png'.format(output_path,i_batch), out)
        print(i_batch)


    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)