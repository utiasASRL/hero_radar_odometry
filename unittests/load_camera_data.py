import argparse
import json
import os
import sys

from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.custom_sampler import *
from datasets.kitti import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/sample.json', type=str,
                      help='config file path (default: config/sample.json)')

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    results_path = "{}/results/unittest/{}".format(config["home_dir"], config["session_name"])
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Print outputs to files
    orig_stdout = sys.stdout
    fl = open('{}/out.txt'.format(results_path), 'w')
    sys.stdout = fl
    orig_stderr = sys.stderr
    fe = open('{}/err.txt'.format(results_path), 'w')
    sys.stderr = fe
    
    # dataloader setup
    train_dataset = KittiDataset(config, 'training')
    train_sampler = RandomWindowBatchSampler(batch_size=config["train_loader"]["batch_size"],
                                             window_size=config["train_loader"]["window_size"],
                                             seq_len=train_dataset.seq_len,
                                             drop_last=True)
    train_loader = DataLoader(train_dataset,
                              batch_sampler=train_sampler,
                              num_workers=config["train_loader"]["num_workers"],
                              pin_memory=True)
    
    to_img = transforms.ToPILImage()

    window_size = config["train_loader"]["window_size"]
    batch_size = config["train_loader"]["batch_size"]

    for i_batch, batch_sample in enumerate(train_loader):
        
        assert batch_sample['input'].size(0) == config["train_loader"]["window_size"] * \
                                                config["train_loader"]["batch_size"]

        # Check the images
        window = 0
        start = window * config["train_loader"]["batch_size"]
        end = start + config["train_loader"]["batch_size"]
        stereo_pair_window0 = batch_sample['input'][start:end, :, :, :]
                
        window = 1
        start = window * config["train_loader"]["batch_size"]
        end = start + config["train_loader"]["batch_size"]
        stereo_pair_window1 = batch_sample['input'][start:end, :, :, :]
        
        batch_size, channels, height, width = stereo_pair_window0.size()

        assert batch_size == config["train_loader"]["batch_size"]
        assert channels == 6
        assert height == config['dataset']['images']['height']
        assert width == config['dataset']['images']['width']

        first_left_img = to_img(stereo_pair_window0[0, :3, :, :]) 
        first_right_img = to_img(stereo_pair_window0[0, 3:, :, :])
        first_left_img.save('{}/first_left_img.png'.format(results_path), 'png')
        first_right_img.save('{}/first_right_img.png'.format(results_path), 'png')

        second_left_img = to_img(stereo_pair_window1[0, :3, :, :])
        second_right_img = to_img(stereo_pair_window1[0, 3:, :, :])
        second_left_img.save('{}/second_left_img.png'.format(results_path))
        second_right_img.save('{}/second_right_img.png'.format(results_path))

        # Check the disparity
        window = 0
        start = window * config["train_loader"]["batch_size"]
        end = start + config["train_loader"]["batch_size"]
        disparity_window0 = batch_sample['geometry'][start:end, :, :, :]
        
        window = 1
        start = window * config["train_loader"]["batch_size"]
        end = start + config["train_loader"]["batch_size"]
        disparity_window1 = batch_sample['geometry'][start:end, :, :, :]

        batch_size, channels, height, width = disparity_window0.size()

        assert batch_size == config["train_loader"]["batch_size"]
        assert channels == 1
        assert height == config['dataset']['images']['height']
        assert width == config['dataset']['images']['width']

        first_disparity = disparity_window0[0, 0, :, :]
        first_disparity[first_disparity < 0.0] = 0.0
        first_disparity = first_disparity / torch.max(first_disparity)
        first_disparity_img = to_img(first_disparity)
        first_disparity_img.save('{}/first_disparity_img.png'.format(results_path), 'png')
        
        second_disparity = disparity_window1[0, 0, :, :]
        second_disparity[second_disparity < 0.0] = 0.0
        second_disparity = second_disparity / torch.max(second_disparity)
        second_disparity_img = to_img(second_disparity)
        second_disparity_img.save('{}/second_disparity_img.png'.format(results_path))

        if i_batch > 3:
            break

    # Stop writing outputs to files.
    sys.stdout = orig_stdout
    fl.close()
    sys.stderr = orig_stderr
    fe.close()
