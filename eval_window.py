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
from utils.window_estimator import WindowEstimator
from utils.window_estimator_pseudo import WindowEstimatorPseudo

# steam
import cpp_wrappers.cpp_steam.build.steampy_f2f as steampy_f2f

# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='results/ir_super3_w6_mah4_p8_win10_00_sobi6v/config.json', type=str,
                      help='config file path (default: config/steam_f2f.json)')

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    config['dataset']['data_dir'] = '/home/david/Data/kitti'

    # Initialize datasets
    test_dataset = KittiDataset(config, set='test')
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)


    # Initialize the dataloader
    test_loader = DataLoader(test_dataset,
                             sampler=test_sampler,
                             num_workers=0,
                             pin_memory=True)

    # gpu
    device = torch.device("cuda:0")

    # load checkpoint
    previous_training_path = config['previous_session']
    chosen_chkp = 'chkp.tar'
    chosen_chkp = os.path.join('results', previous_training_path, chosen_chkp)
    checkpoint = torch.load(chosen_chkp)

    with torch.no_grad():

        # network
        net = F2FPoseModel(config,
                             config['test_loader']['window_size'],
                             config['test_loader']['batch_size'])
        net.to(device)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()

        T_k0_steam = np.zeros((test_dataset.seq_len[0], 4, 4))
        T_21_steam = np.zeros((test_dataset.seq_len[0] - 1, 4, 4))
        T_k0_gt = np.zeros((test_dataset.seq_len[0], 4, 4))
        T_21_gt = np.zeros((test_dataset.seq_len[0] - 1, 4, 4))

        # TODO: FIX THIS
        net.keypoint_block.window_size = 1

        # window optimizer
        # solver = WindowEstimator()
        solver = WindowEstimatorPseudo(config['test_loader']['window_size'], config['networks']['min_obs'], config['networks']['pseudo_temp'],
                                       config['networks']['keypoint_loss']['mah'])

        # load random pair
        for i_batch, sample_batch in enumerate(test_loader):
            fid = int(sample_batch['f_ind'][0])

            coords, descs, weights = net.forward_keypoints_single(sample_batch)

            # add new frame
            solver.add_frame(coords, descs, weights)
            if solver.isWindowFull():
                # optimize
                solver.optimize()

                # save out first frame
                T_k0, out_id = solver.getFirstPose()
                T_k0_steam[out_id, :, :] = T_k0

                if out_id > 0:
                    # compose
                    T_21_steam[out_id-1, :, :] = T_k0_steam[out_id, :, :]@\
                                                 net.se3_inv(torch.from_numpy(T_k0_steam[out_id-1, :, :])).numpy()

            # groundtruth
            T_iv = sample_batch['T_iv']
            T_k0_gt[fid, :, :] = net.se3_inv(T_iv[0, :, :]).numpy()
            if fid > 0:
                T_21_gt[fid-1, :, :] = T_k0_gt[fid, :, :]@\
                                       net.se3_inv(torch.from_numpy(T_k0_gt[fid-1, :, :])).numpy()
            print(fid)

            # save periodically
            if np.mod(fid, 200) == 0 and fid != 0:
                np.savetxt('traj/T_21_steam.txt', np.reshape(T_21_steam[:out_id, :, :], (-1, 4)), delimiter=',')
                np.savetxt('traj/T_21_gt.txt', np.reshape(T_21_gt[:out_id, :, :], (-1, 4)), delimiter=',')
                a = 1
                # np.savetxt('traj/T_21_ransac.txt', np.reshape(T_21_ransac[:out_id, :, :], (-1, 4)), delimiter=',')

        # np.savetxt('traj/T_21_steam.txt', np.reshape(T_21_steam, (-1, 4)), delimiter=',')
        # np.savetxt('traj/T_21_gt.txt', np.reshape(T_21_gt, (-1, 4)), delimiter=',')
        # np.savetxt('traj/T_21_ransac.txt', np.reshape(T_21_ransac, (-1, 4)), delimiter=',')

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)