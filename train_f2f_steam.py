import os
import signal
import argparse
import json

import torch.nn as nn
from torch.utils.data import DataLoader

from utils.trainer import Trainer
from datasets.custom_sampler import *
from datasets.kitti import *
from networks.f2f_pose_model import F2FPoseModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/steam_f2f.json', type=str,
                      help='config file path (default: config/sample.json)')

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

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
    model = F2FPoseModel(config,
                         config['train_loader']['window_size'],
                         config['train_loader']['batch_size'])

    # trainer
    trainer = Trainer(model, train_loader, valid_loader, config)

    # train
    trainer.train()

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)

