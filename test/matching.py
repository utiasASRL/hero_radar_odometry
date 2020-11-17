import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pickle
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.custom_sampler import *
from datasets.kitti import *
from datasets.custom_sampler import RandomWindowBatchSampler
from networks.svd_pose_model import SVDPoseModel
from utils.lie_algebra import so3_to_rpy
from utils.plot import plot_route, plot_versus, plot_error
from utils.stereo_camera_model import StereoCameraModel

class MatchTester:
    """
    MatchTester class
    """
    def __init__(self, model, train_loader, test_loader, config, result_path):
        # network
        self.model = model

        # move the network to GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

        # load network parameters and optimizer if resuming from previous session
        assert config['previous_session'] != "", "No previous session checkpoint provided!"

        checkpoint_path = "{}/{}/{}/{}".format(config['home_dir'], 'results', config['previous_session'], 'checkpoints/chkp.tar')
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print("Model and training state restored.")

        # data loaders
        self.train_loader = train_loader
        self.test_loader = test_loader

        # config dictionary
        self.config = config
        self.window_size = config['test_loader']['window_size']

        # stereo camera model
        self.stereo_cam = StereoCameraModel()

        # logging
        self.result_path = result_path
        self.plot_path = '{}/{}'.format(self.result_path, 'visualization')
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)

    def visualize_matches(self, images, disparity, cam_calib, saved_data, data_type, index):

        src_coords = saved_data['src_coords']
        src_2D = saved_data['src_2D']
        pseudo_2D = saved_data['pseudo_2D']
        pseudo_gt_2D = saved_data['pseudo_gt_2D']
        weight_scores_src = saved_data['weight_scores_src']
        weight_scores_tgt = saved_data['weight_scores_tgt']
        inliers = saved_data['inliers']
        weights = saved_data['weights'] if (config['loss']['start_svd_epoch'] == 0) else torch.ones(inliers.size())
        src_valid = saved_data['src_valid']
        pseudo_valid = saved_data['pseudo_valid']

        # Test that stereo camera projection works correctly.
        src_project_2D = self.stereo_cam.camera_model(src_coords, cam_calib,
                                                      start_ind=0, step=2)[:, 0:2, :].transpose(2, 1)  # BxNx2

        # Create images that we will use for visualization
        to_img = transforms.ToPILImage()
        img_src_tensor = images[0, 0:3, :, :].detach().cpu()
        img_src = to_img(img_src_tensor)
        img_tgt_tensor = images[1, 0:3, :, :].detach().cpu()
        img_tgt = to_img(img_tgt_tensor)

        img_motion = to_img(img_tgt_tensor)
        img_motion_gt = to_img(img_tgt_tensor)
        img_motion_inl = to_img(img_tgt_tensor)
        img_motion_err_inl = to_img(img_tgt_tensor)
        img_motion_err_weight = to_img(img_tgt_tensor)

        disp_src = disparity[0, :, :].unsqueeze(0).detach().cpu()
        disp_tgt = disparity[1, :, :].unsqueeze(0).detach().cpu()
        disp_src[disp_src < 0.0] = 0.0
        disp_tgt[disp_tgt < 0.0] = 0.0
        disp_src = disp_src / torch.max(disp_src)
        disp_tgt = disp_tgt / torch.max(disp_tgt)
        img_disp_src = to_img(disp_src)
        img_disp_tgt = to_img(disp_tgt)

        # if not config['network']['args']['orb']:
        #     descriptor_match = DescriptorMatchDebug(results_folder)
        #     descriptor_match(features_src, features_tgt, "{}_{}".format(path_ind, count), 'all')

        draw_src = ImageDraw.Draw(img_src)
        draw_motion = ImageDraw.Draw(img_motion)
        draw_motion_gt = ImageDraw.Draw(img_motion_gt)
        draw_motion_inl = ImageDraw.Draw(img_motion_inl)
        draw_motion_err_inl = ImageDraw.Draw(img_motion_err_inl)
        draw_motion_err_weight = ImageDraw.Draw(img_motion_err_weight)
        draw_disp_src = ImageDraw.Draw(img_disp_src)
        draw_disp_tgt = ImageDraw.Draw(img_disp_tgt)

        for i in range(0, src_2D.size(1)):

            x_src = src_2D[0, i, 0]
            y_src = src_2D[0, i, 1]

            x_tgt = pseudo_2D[0, i, 0]
            y_tgt = pseudo_2D[0, i, 1]

            x_tgt_gt = pseudo_gt_2D[0, i, 0]
            y_tgt_gt = pseudo_gt_2D[0, i, 1]

            x_src_prj = src_project_2D[0, i, 0]
            y_src_prj = src_project_2D[0, i, 1]

            colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            if (src_valid[0, 0, i] != 0) and (pseudo_valid[0, 0, i] != 0):

                draw_src.ellipse([x_src - 3.0, y_src - 3.0, x_src + 3.0, y_src + 3.0], outline=colour)
                draw_src.ellipse([x_src_prj - 3.0, y_src_prj - 3.0, x_src_prj + 3.0, y_src_prj + 3.0], fill=colour)
                draw_src.line([x_src, y_src, x_src_prj, y_src_prj], fill=colour)

                draw_motion.ellipse([x_src - 3.0, y_src - 3.0, x_src + 3.0, y_src + 3.0], fill=colour, outline=colour)
                draw_motion.ellipse([x_tgt - 3.0, y_tgt - 3.0, x_tgt + 3.0, y_tgt + 3.0], outline=colour)
                draw_motion.line([x_src, y_src, x_tgt, y_tgt], fill=colour)

                draw_motion_gt.ellipse([x_src - 3.0, y_src - 3.0, x_src + 3.0, y_src + 3.0], fill=colour, outline=colour)
                draw_motion_gt.ellipse([x_tgt_gt - 3.0, y_tgt_gt - 3.0, x_tgt_gt + 3.0, y_tgt_gt + 3.0], outline=colour)
                draw_motion_gt.line([x_src, y_src, x_tgt_gt, y_tgt_gt], fill=colour)

                if inliers[0, 0, i] != 0.0:
                    draw_motion_inl.ellipse([x_src - 3.0, y_src - 3.0, x_src + 3.0, y_src + 3.0], fill=(0, 255, 0),
                                            outline=(0, 255, 0))
                    draw_motion_inl.ellipse([x_tgt - 3.0, y_tgt - 3.0, x_tgt + 3.0, y_tgt + 3.0], outline=(0, 255, 0))
                    draw_motion_inl.line([x_src, y_src, x_tgt, y_tgt], fill=(0, 255, 0))

                    draw_motion_err_inl.line([x_tgt, y_tgt, x_tgt_gt, y_tgt_gt], fill=(0, 255, 0), width=2)
                else:
                    draw_motion_inl.ellipse([x_src - 3.0, y_src - 3.0, x_src + 3.0, y_src + 3.0], fill=(255, 0, 0),
                                            outline=(255, 0, 0))
                    draw_motion_inl.ellipse([x_tgt - 3.0, y_tgt - 3.0, x_tgt + 3.0, y_tgt + 3.0], outline=(255, 0, 0))
                    draw_motion_inl.line([x_src, y_src, x_tgt, y_tgt], fill=(255, 0, 0))

                    draw_motion_err_inl.line([x_tgt, y_tgt, x_tgt_gt, y_tgt_gt], fill=(255, 0, 0), width=2)

                if weights[0, 0, i] != 0.0:
                    draw_motion_err_weight.line([x_tgt, y_tgt, x_tgt_gt, y_tgt_gt], fill=(0, 255, 0), width=2)
                else:
                    draw_motion_err_weight.line([x_tgt, y_tgt, x_tgt_gt, y_tgt_gt], fill=(255, 0, 0), width=2)

                draw_disp_src.ellipse([x_src - 3.0, y_src - 3.0, x_src + 3.0, y_src + 3.0], fill=128)
                draw_disp_tgt.ellipse([x_tgt - 3.0, y_tgt - 3.0, x_tgt + 3.0, y_tgt + 3.0], fill=128)

            else:

                if src_valid[0, 0, i] != 0:
                    draw_disp_src.ellipse([x_src - 3.0, y_src - 3.0, x_src + 3.0, y_src + 3.0], fill=128)
                else:
                    draw_disp_src.ellipse([x_src - 3.0, y_src - 3.0, x_src + 3.0, y_src + 3.0], outline=128)

                if pseudo_valid[0, 0, i] != 0:
                    draw_disp_tgt.ellipse([x_tgt - 3.0, y_tgt - 3.0, x_tgt + 3.0, y_tgt + 3.0], fill=128)
                else:
                    draw_disp_tgt.ellipse([x_tgt - 3.0, y_tgt - 3.0, x_tgt + 3.0, y_tgt + 3.0], outline=128)

        del draw_src
        del draw_motion
        del draw_motion_gt
        del draw_motion_inl
        del draw_motion_err_inl
        del draw_motion_err_weight
        del draw_disp_src
        del draw_disp_tgt

        img_src.save("{}/img_src_{}_{}.png".format(self.plot_path, data_type, index), "png")
        img_motion.save("{}/img_motion_{}_{}.png".format(self.plot_path, data_type, index), "png")
        img_motion_gt.save("{}/img_motion_gt_{}_{}.png".format(self.plot_path, data_type, index), "png")
        img_motion_inl.save("{}/img_motion_inl_{}_{}.png".format(self.plot_path, data_type, index), "png")
        img_motion_err_inl.save("{}/img_motion_err_inl_{}_{}.png".format(self.plot_path, data_type, index), "png")
        img_motion_err_weight.save("{}/img_motion_err_weight_{}_{}.png".format(self.plot_path, data_type, index), "png")
        img_disp_src.save("{}/img_disp_src_{}_{}.png".format(self.plot_path, data_type, index), "png")
        img_disp_tgt.save("{}/img_disp_tgt_{}_{}.png".format(self.plot_path, data_type, index), "png")

        # Store the learned score images
        score_img_src = to_img(weight_scores_src[0, :, :, :].detach().cpu())
        score_img_tgt = to_img(weight_scores_tgt[0, :, :, :].detach().cpu())

        score_img_src.save("{}/weight_score_img_src_{}_{}.png".format(self.plot_path, data_type, index), "png")
        score_img_tgt.save("{}/weight_score_img_tgt_{}_{}.png".format(self.plot_path, data_type, index), "png")

    def test_epoch(self, data_loader, data_type):
        self.model.eval()

        with torch.no_grad():
            for i_batch, batch_sample in enumerate(data_loader):

                if i_batch > 5:
                    break

                # forward prop
                try:
                    loss = self.model(batch_sample, 0)
                except Exception as e:
                    self.model.print_loss(loss, 0, i_batch)
                    self.model.print_inliers(0, i_batch)
                    print(e)

                self.model.print_inliers(0, i_batch)
                print("loss:{}".format(loss))

                # collect data
                save_dict = self.model.return_save_dict()

                self.visualize_matches(batch_sample['input'],
                                       batch_sample['geometry'],
                                       batch_sample['cam_calib'],
                                       save_dict, data_type, i_batch)

    def test(self):
        """
        Full testing loop
        """

        self.test_epoch(self.train_loader, 'train')
        self.test_epoch(self.test_loader, 'test')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/sample.json', type=str,
                        help='config file path (default: config/sample.json)')

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    result_path = os.path.join(config['home_dir'], 'results', config['session_name'])
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # logging
    # stdout_orig = sys.stdout
    # log_path_out = os.path.join(result_path, 'out_train.txt')
    # stdout_file = open(log_path_out, 'w')
    # sys.stdout = stdout_file
    #
    # stderr_orig = sys.stderr
    # log_path_err = os.path.join(result_path, 'err_train.txt')
    # stderr_file = open(log_path_err, 'w')
    # sys.stderr = stderr_file

    # dataloader setup
    train_dataset = KittiDataset(config, set='training')
    train_sampler = RandomWindowBatchSampler(batch_size=config["test_loader"]["batch_size"],
                                                window_size=config["test_loader"]["window_size"],
                                                seq_len=train_dataset.seq_len,
                                                drop_last=True)
    train_loader = DataLoader(train_dataset,
                             batch_sampler=train_sampler,
                             num_workers=config["test_loader"]["num_workers"],
                             pin_memory=True)

    test_dataset = KittiDataset(config, set='test')
    test_sampler = RandomWindowBatchSampler(batch_size=config["test_loader"]["batch_size"],
                                                window_size=config["test_loader"]["window_size"],
                                                seq_len=test_dataset.seq_len,
                                                drop_last=True)
    test_loader = DataLoader(test_dataset,
                             batch_sampler=test_sampler,
                             num_workers=config["test_loader"]["num_workers"],
                             pin_memory=True)

    # network setup
    model = SVDPoseModel(config,
                         config['test_loader']['window_size'],
                         config['test_loader']['batch_size'])

    # trainer
    match_tester = MatchTester(model, train_loader, test_loader, config, result_path)

    # train
    match_tester.test()

    # stop writing outputs to files
    # sys.stdout = stdout_orig
    # stdout_file.close()
    # sys.stderr = stderr_orig
    # stderr_file.close()