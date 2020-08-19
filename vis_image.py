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
    parser.add_argument('--config', default='config/steam_f2f.json', type=str,
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
    # checkpoint = torch.load(chosen_chkp)

    # set output path
    output_path = os.path.join('plot', previous_training_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # network
    net = F2FPoseModel(config,
                         config['test_loader']['window_size'],
                         config['test_loader']['batch_size'])
    net.to(device)
    # net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()


    # load random pair
    for i_batch, data in enumerate(training_loader):

        # parse data
        geometry_img, images, T_iv = data['geometry'], data['input'], data['T_iv']

        hborder = np.ones((10, images.size(3))).astype(np.float32)*0.1

        # move to GPU
        geometry_img = geometry_img.cuda()
        images = images.cuda()
        T_iv = T_iv.cuda()

        # forward pass
        # detector_scores, weight_scores, descs = net.forward_encoder_decoder(images)

        batch_id = 0
        # intensity/range image
        intensity_im = images[batch_id, 0, :, :].detach().cpu().numpy()
        # range_im = images[batch_id, 1, :, :].detach().cpu().numpy()
        # ir_im = np.concatenate([intensity_im, range_im], axis=0)
        ir_im = intensity_im
        # plt.imsave('{}/intensity_range{}.png'.format(output_path, i_batch), ir_im)

        # intensity + score map
        # intensity_im = sample_batch['image'][batch_id, 4, :, :].detach().numpy()
        # score_im = np.exp(weight_scores[batch_id, 0, :, :].detach().cpu().numpy())
        # mind = np.min(score_im)
        # maxd = np.max(score_im)
        # score_im = (score_im-mind)/(maxd-mind)
        # score_im = score_im/np.max(score_im)
        # out = np.concatenate([intensity_im, score_im], axis=0)
        # plt.imsave('{}/weight{}.png'.format(output_path,i_batch), score_im)

        # detector
        # detect_im = detector_scores[batch_id, 0, :, :].detach().cpu().numpy()
        # mind = np.min(detect_im)
        # maxd = np.max(detect_im)
        # detect_im = (detect_im-mind)/(maxd-mind)
        # out = np.concatenate([ir_im, detect_im, score_im], axis=0)
        # out = np.concatenate([detect_im, score_im], axis=0)
        # out = detect_im
        # plt.imsave('{}/detect_weight{}.png'.format(output_path,i_batch), out)


        # sobel
        bsobelx, bsobely = net.sobel(images[:, 0:1, :, :])
        int_sobel_x = bsobelx[batch_id, 0, :, :].detach().cpu().numpy()
        int_sobel_y = bsobely[batch_id, 0, :, :].detach().cpu().numpy()
        # out = np.concatenate(np.abs([int_sobel_x, int_sobel_y]), axis=0)
        # plt.imsave('{}/int_sobel{}.png'.format(output_path,i_batch), out)

        bsobelx, bsobely = net.sobel(images[:, 1:2, :, :])
        ran_sobel_x = bsobelx[batch_id, 0, :, :].detach().cpu().numpy()
        ran_sobel_y = bsobely[batch_id, 0, :, :].detach().cpu().numpy()
        # out = np.concatenate(np.abs([ran_sobel_x, ran_sobel_y]), axis=0)
        # plt.imsave('{}/ran_sobel{}.png'.format(output_path,i_batch), out)

        # bsobelx, bsobely = net.sobel(images)
        # int_sobel_x = bsobelx[batch_id, 0, :, :].detach().cpu().numpy()
        # int_sobel_y = bsobely[batch_id, 0, :, :].detach().cpu().numpy()
        # ran_sobel_x = bsobelx[batch_id, 1, :, :].detach().cpu().numpy()
        # ran_sobel_y = bsobely[batch_id, 1, :, :].detach().cpu().numpy()
        # out = np.concatenate([int_sobel_x, int_sobel_y, ran_sobel_x, ran_sobel_y], axis=0)
        # plt.imsave('{}/sobel{}.png'.format(output_path,i_batch), out)

        # vehicle mask
        vehicle_mask = np.ones((images.size(2), images.size(3)))

        # left/right vertical border
        vborder = 6
        vehicle_mask[:, :vborder+1] = 0.2
        vehicle_mask[:, -vborder:] = 0.2

        # bottom-left blob
        vehicle_mask[-23:, :48] = 0.5
        vehicle_mask[-35:, 24:54] = 0.5
        vehicle_mask[-15:, 48:82] = 0.5
        vehicle_mask[-23:, 72:112] = 0.5

        # bottom-centre blob
        vehicle_mask[-17:, 275:440] = 0.5

        # bottom-centre-right blob
        vehicle_mask[-27:, 460:533] = 0.5
        vehicle_mask[-17:, 533:541] = 0.5

        # bottom-right blob
        # vehicle_mask[-8:, 580:590] = 0.5
        vehicle_mask[-20:, 580:655] = 0.5
        vehicle_mask[-32:, 655:695] = 0.5
        vehicle_mask[-22:, 695:] = 0.5

        # sobel mask
        thresh = 0.6
        out = np.concatenate([intensity_im, hborder, np.abs(int_sobel_x) > thresh, np.abs(int_sobel_y) > thresh,
                              hborder, (np.abs(int_sobel_x) > thresh)*vehicle_mask, hborder, vehicle_mask], axis=0)
        plt.imsave('{}/int_sobel_mask{}.png'.format(output_path, i_batch), out)

        thresh = 0.2
        # out = np.concatenate(np.abs([ran_sobel_x, ran_sobel_y]) > thresh, axis=0)
        mag = np.sqrt(ran_sobel_x ** 2 + ran_sobel_y ** 2)

        out = np.concatenate([intensity_im, hborder, mag > thresh, hborder, (mag > thresh)*vehicle_mask,
                              hborder, vehicle_mask], axis=0)
        plt.imsave('{}/ran_sobel_mask{}.png'.format(output_path,i_batch), out)

        # loss
        # loss = net.loss(keypoints_3D, keypoints_scores, T_iv, pseudo_ref)
        print(i_batch)



    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)