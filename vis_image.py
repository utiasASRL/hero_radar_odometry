import signal
import os
from os import makedirs, remove
from os.path import exists, join
import time
import argparse
import json
import matplotlib.pyplot as plt

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
    parser.add_argument('--config', default='config/steam_f2f_eval.json', type=str,
                      help='config file path (default: config/steam_f2f.json)')

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # Initialize datasets
    train_dataset = KittiDataset(config, set='training')
    # train_sampler = RandomWindowBatchSampler(batch_size=1,
    #                                          window_size=2,
    #                                          seq_len=train_dataset.seq_len,
    #                                          drop_last=True)
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
                         config['train_loader']['window_size'],
                         config['train_loader']['batch_size'])
    net.to(device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()


    # load random pair
    for i_batch, data in enumerate(training_loader):

        # parse data
        geometry_img, images, T_iv = data['geometry'], data['input'], data['T_iv']

        # move to GPU
        geometry_img = geometry_img.cuda()
        images = images.cuda()
        T_iv = T_iv.cuda()

        # forward pass
        detector_scores, weight_scores, descs = net.forward_encoder_decoder(images)

        batch_id = 0
        # intensity/range image
        intensity_im = images[batch_id, 0, :, :].detach().cpu().numpy()
        range_im = images[batch_id, 1, :, :].detach().cpu().numpy()
        ir_im = np.concatenate([intensity_im, range_im], axis=0)
        # plt.imsave('{}/intensity_range{}.png'.format(output_path, i_batch), ir_im)

        # intensity + score map
        # intensity_im = sample_batch['image'][batch_id, 4, :, :].detach().numpy()
        score_im = np.exp(weight_scores[batch_id, 0, :, :].detach().cpu().numpy())
        mind = np.min(score_im)
        maxd = np.max(score_im)
        score_im = (score_im-mind)/(maxd-mind)
        # score_im = score_im/np.max(score_im)
        # out = np.concatenate([intensity_im, score_im], axis=0)
        # plt.imsave('{}/weight{}.png'.format(output_path,i_batch), score_im)

        # detector
        detect_im = detector_scores[batch_id, 0, :, :].detach().cpu().numpy()
        mind = np.min(detect_im)
        maxd = np.max(detect_im)
        detect_im = (detect_im-mind)/(maxd-mind)
        out = np.concatenate([ir_im, detect_im, score_im], axis=0)
        plt.imsave('{}/detect_weight{}.png'.format(output_path,i_batch), out)
        print(i_batch)


        # loss
        # loss = net.loss(keypoints_3D, keypoints_scores, T_iv, pseudo_ref)




    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)