import signal
import argparse
import json
import os
from os import makedirs, remove
from os.path import exists, join
import time

# Dataset
from datasets.custom_sampler import *
from datasets.kitti import *
from utils.config import *
from torch.utils.data import DataLoader
import torch.nn.functional as F

# network
from networks.f2f_pose_model import F2FPoseModel

# steam
import cpp_wrappers.cpp_steam.build.steampy_f2f as steampy_f2f

# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/steam_f2f_eval.json', type=str,
                      help='config file path (default: config/steam_f2f.json)')

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # Initialize datasets
    train_dataset = KittiDataset(config, set='training')
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
    # chkp_path = os.path.join('results', previous_training_path, 'checkpoints')
    chosen_chkp = 'chkp.tar'
    chosen_chkp = os.path.join('results', previous_training_path, chosen_chkp)
    checkpoint = torch.load(chosen_chkp)

    # network
    net = F2FPoseModel(config,
                         config['train_loader']['window_size'],
                         config['train_loader']['batch_size'])
    net.to(device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    T_21_steam = np.zeros((train_dataset.seq_len[0] - 1, 4, 4))
    T_21_gt = np.zeros((train_dataset.seq_len[0] - 1, 4, 4))

    # load random pair
    for i_batch, sample_batch in enumerate(training_loader):
        fid = int(sample_batch['f_ind'][0])

        src_coords, tgt_coords, weights = net.forward_keypoints(sample_batch)

        # get src points
        points1 = src_coords[0, :, :].transpose(0, 1)

        # check for no returns in src keypoints
        nr_ids = torch.nonzero(torch.sum(points1, dim=1), as_tuple=False).squeeze()
        points1 = points1[nr_ids, :]

        # get tgt points
        points2 = tgt_coords[0, :, nr_ids].transpose(0, 1)

        # get weights
        w = weights[0, :, nr_ids].transpose(0, 1)

        # match consistency
        w12 = net.softmax_matcher_block.match_vals[0, nr_ids, :] # N x M
        # w12 = torch.zeros((5, 4))
        _, ind2to1 = torch.max(w12, dim=1)  # N
        _, ind1to2 = torch.max(w12, dim=0)  # M
        mask = torch.eq(ind1to2[ind2to1], torch.arange(ind2to1.__len__(), device=ind1to2.device))
        mask_ind = torch.nonzero(mask, as_tuple=False).squeeze()
        points1 = points1[mask_ind, :]
        points2 = points2[mask_ind, :]
        w = w[mask_ind, :]

        points2 = points2[:, :].detach().cpu().numpy()
        points1 = points1[:, :].detach().cpu().numpy()

        D = torch.zeros(w.size(0), 9, device=w.device)
        D[:, (0, 4, 8)] = torch.exp(w)
        D = D.reshape((-1, 3, 3))
        D = D.detach().cpu().numpy()

        # steam
        T_21_temp = np.zeros((13, 4, 4), dtype=np.float32)
        steampy_f2f.run_steam_best_match(points1, points2, D, T_21_temp)
        T_21_steam[fid, :, :] = T_21_temp[0, :, :]

        # gt
        T_iv = sample_batch['T_iv']
        T_21 = net.se3_inv(T_iv[1, :, :])@T_iv[0, :, :]
        T_21_gt[fid, :, :] = T_21.numpy()
        print(fid)

        # save periodically
        if np.mod(fid, 200) == 0:
            np.savetxt('traj/T_21_steam.txt', np.reshape(T_21_steam[:fid, :, :], (-1, 4)), delimiter=',')
            np.savetxt('traj/T_21_gt.txt', np.reshape(T_21_gt[:fid, :, :], (-1, 4)), delimiter=',')

    np.savetxt('traj/T_21_steam.txt', np.reshape(T_21_steam, (-1, 4)), delimiter=',')
    np.savetxt('traj/T_21_gt.txt', np.reshape(T_21_gt, (-1, 4)), delimiter=',')

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)