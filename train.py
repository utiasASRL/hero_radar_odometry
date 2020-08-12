import os
import signal
import argparse
import json

import torch.nn as nn
from torch.utils.data import DataLoader

from utils.trainer import Trainer
from datasets.custom_sampler import *
from datasets.kitti import *
from networks.svd_pose_model import SVDPoseModel

def _init_saving(args):
    # parse args
    config_fname = args.config

    # create directories
    result_path = 'results/' + config['session_name']
    chkp_dir = os.path.join(result_path, 'checkpoints')
    if not os.path.exists(chkp_dir):
        os.makedirs(chkp_dir)
    if not os.path.exists('{}/backup'.format(chkp_dir)):
        os.makedirs('{}/backup'.format(chkp_dir))

    os.system('cp {} {}/backup/config.json.backup'.format(config_fname, chkp_dir))
    os.system('cp networks/svd_pose_model.py {}/backup/svd_pose_model.py.backup'.format(chkp_dir))
    os.system('cp networks/unet_block.py {}/backup/unet_block.py.backup'.format(chkp_dir))
    os.system('cp networks/keypoint_block.py {}/backup/keypoint_block.py.backup'.format(chkp_dir))
    os.system('cp networks/softmax_matcher_block.py {}/backup/softmax_matcher_block.py.backup'.format(chkp_dir))
    os.system('cp networks/svd_weight_block.py {}/backup/svd_weight_block.py.backup'.format(chkp_dir))
    os.system('cp networks/svd_block.py {}/backup/svd_block.py.backup'.format(chkp_dir))
    os.system('cp networks/layers.py {}/backup/layers.py.backup'.format(chkp_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/sample.json', type=str,
                      help='config file path (default: config/sample.json)')
    parser.add_argument('--session_name', type=str, default='session', metavar='N',
                        help='Session name')

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # set session name depending on input arg
    config["session_name"] = args.session_name

    # save copies of important files
    _init_saving(args)

    # dataloader setup
    train_dataset = KittiDataset(config, set='training')
    train_sampler = RandomWindowBatchSampler(batch_size=config["train_loader"]["batch_size"],
                                             window_size=config["train_loader"]["window_size"],
                                             seq_len=train_dataset.seq_len,
                                             drop_last=True)
    train_loader = DataLoader(train_dataset,
                              batch_sampler=train_sampler,
                              num_workers=config["train_loader"]["num_workers"],
                              pin_memory=True)

    valid_dataset = KittiDataset(config, set='validation')
    valid_sampler = RandomWindowBatchSampler(batch_size=config["train_loader"]["batch_size"],
                                             window_size=config["train_loader"]["window_size"],
                                             seq_len=valid_dataset.seq_len,
                                             drop_last=True)
    valid_loader = DataLoader(train_dataset,
                              batch_sampler=valid_sampler,
                              num_workers=config["train_loader"]["num_workers"],
                              pin_memory=True)

    # network setup
    model = SVDPoseModel(config,
                         config['train_loader']['window_size'],
                         config['train_loader']['batch_size'])

    # trainer
    trainer = Trainer(model, train_loader, valid_loader, config)

    # train
    trainer.train()

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)

