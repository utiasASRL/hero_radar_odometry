#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to visualize on TrajKittiP2P dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      David Yoon - 19/04/2020
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# sys imports
import os
import sys
import signal
import argparse
import json

# third-party imports
import numpy as np
from torch.utils.data import DataLoader

# project imports
from datasets.custom_sampler import *
from datasets.kitti import *
from utils.config import *
from networks.svd_pose_model import SVDPoseModel
from utils.trainer import Trainer

# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ############################
    # Initialize the environment
    ############################

    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--config', default='config/sample.json', type=str,
                        help='config file path (default: config/sample.json)')
    parser.add_argument('--session_name', type=str, default='session', metavar='N',
                        help='Session name')
    parser.add_argument('--job_name', type=str, default='exp', metavar='N',
                        help='Name of the job to evaluate')
    parser.add_argument('--prev_chkp', type=str, default='', metavar='N',
                        help='Previous checkpoint')

    args = parser.parse_args()

    # Set which gpu is going to be used
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    seed = 1234
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(seed)

    ###############
    # Previous chkp
    ###############

    # Choose here if you want to start training from a previous snapshot (None for new training)
    # previous_training_path = 'Log_2020-03-19_19-53-27'
    previous_training_path = args.job_name

    # Choose index of checkpoint to start from. If None, uses the latest chkp
    chkp_idx = None
    if previous_training_path:

        # Find all snapshot in the chosen training folder
        chkp_path = os.path.join('results', previous_training_path, 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

        # Find which snapshot to restore
        if chkp_idx is None:
            chosen_chkp = 'chkp.tar'
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join('results', previous_training_path, 'checkpoints', chosen_chkp)

    else:
        chosen_chkp = None

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    # Initialize configuration class
    with open(args.config) as f:
        config = json.load(f)

    # set session name depending on input arg
    config["session_name"] = args.session_name

    # Initialize datasets
    train_dataset = KittiDataset(config, set='training')
    valid_dataset = KittiDataset(config, set='validation')

    # Initialize samplers
    train_sampler = RandomWindowBatchSampler(batch_size=config["train_loader"]["batch_size"],
                                             window_size=config["train_loader"]["window_size"],
                                             seq_len=train_dataset.seq_len,
                                             drop_last=True)
    valid_sampler = RandomWindowBatchSampler(batch_size=config["train_loader"]["batch_size"],
                                             window_size=config["train_loader"]["window_size"],
                                             seq_len=valid_dataset.seq_len,
                                             drop_last=True)

    # Initialize the dataloader
    train_loader = DataLoader(train_dataset,
                              batch_sampler=train_sampler,
                              num_workers=config["train_loader"]["num_workers"],
                              pin_memory=True)
    valid_loader = DataLoader(train_dataset,
                              batch_sampler=valid_sampler,
                              num_workers=config["train_loader"]["num_workers"],
                              pin_memory=True)
    print('*****************')

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    model = SVDPoseModel(config,
                         config['train_loader']['window_size'],
                         config['train_loader']['batch_size'])
    device = torch.device("cuda:0")
    model.to(device)

    # load network
    checkpoint = torch.load(chosen_chkp)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model restored and ready for testing.")

    print('\nLoad one instance of data for testing')
    data = next(iter(train_loader))
    # data = next(iter(validation_loader))

    # pickle load
    # with open('batch_query_0-3d', 'rb') as batch_query_file:
    #     batch = pickle.load(batch_query_file)

    # # perturb pointcloud
    # # Choose two random angles for the first vector in polar coordinates
    # theta = np.random.rand() * 2 * np.pi
    # phi = (np.random.rand() - 0.5) * np.pi
    #
    # # Create the first vector in carthesian coordinates
    # u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])
    #
    # # Choose a random rotation angle
    # alpha = np.random.rand() * 2 * np.pi
    #
    # # Create the rotation matrix with this vector and angle
    # R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]
    # R = R.astype(np.float32)
    # temp = batch[1].features[:, 1:4]@torch.tensor(R.T) + (10*np.random.rand(1, 3) - 5)
    # batch[1].features[:, 1:4] = temp

    # forward
    loss = model(data)
    model.save_intermediate_outputs()
    print(loss)
    a = 1

    # # get pointclouds
    # R_pred = outputs['R_pred'].detach().cpu().numpy()
    # t_pred = outputs['t_pred'].detach().cpu().numpy()
    # T_iv = outputs['T_iv'].detach().cpu().numpy()
    # keypoints_2D_src = outputs['keypoints_2D_src'].detach().cpu().numpy()
    # keypoints_2D_pseudo = outputs['keypoints_2D_pseudo'].detach().cpu().numpy()
    # keypoints_2D_gt = outputs['keypoints_2D_gt'].detach().cpu().numpy()
    # scores_pts_src = outputs['scores_pts_src'].detach().cpu().numpy()
    # scores_pts_trg = outputs['scores_pts_trg'].detach().cpu().numpy()
    # match_vals = outputs['match_vals'].detach().cpu().numpy()
    # match_val_pos = outputs['match_val_pos'].detach().cpu().numpy()
    # keypoints_3D_src = outputs['keypoints_3D_src'].detach().cpu().numpy()
    # keypoints_3D_pseudo = outputs['keypoints_3D_pseudo'].detach().cpu().numpy()
    # images = outputs['images'].detach().cpu().numpy()
    # scores_src = outputs['scores_src'].detach().cpu().numpy()
    # scores_trg = outputs['scores_trg'].detach().cpu().numpy()
    #
    # # save
    # save_dir = os.path.join(config.saving_path)
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    # np.save('{}/R_pred.npy'.format(save_dir), R_pred)
    # np.save('{}/t_pred.npy'.format(save_dir), t_pred)
    # np.save('{}/T_iv.npy'.format(save_dir), T_iv)
    # np.save('{}/keypoints_2D_src.npy'.format(save_dir), keypoints_2D_src)
    # np.save('{}/keypoints_2D_pseudo.npy'.format(save_dir), keypoints_2D_pseudo)
    # np.save('{}/keypoints_2D_gt.npy'.format(save_dir), keypoints_2D_gt)
    # np.save('{}/scores_pts_src.npy'.format(save_dir), scores_pts_src)
    # np.save('{}/scores_pts_trg.npy'.format(save_dir), scores_pts_trg)
    # np.save('{}/match_vals.npy'.format(save_dir), match_vals)
    # np.save('{}/match_val_pos.npy'.format(save_dir), match_val_pos)
    # np.save('{}/keypoints_3D_src.npy'.format(save_dir), keypoints_3D_src)
    # np.save('{}/keypoints_3D_pseudo.npy'.format(save_dir), keypoints_3D_pseudo)
    # np.save('{}/images.npy'.format(save_dir), images)
    # np.save('{}/scores_src.npy'.format(save_dir), scores_src)
    # np.save('{}/scores_trg.npy'.format(save_dir), scores_trg)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)