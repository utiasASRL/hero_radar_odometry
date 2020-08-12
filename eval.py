import os
import signal
import argparse
import json

import torch.nn as nn
from torch.utils.data import DataLoader

from utils.tester import Tester
from datasets.sequential_sampler import SequentialWindowBatchSampler
from datasets.kitti import *
from networks.svd_pose_model import SVDPoseModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/sample.json', type=str,
                        help='config file path (default: config/sample.json)')
    parser.add_argument('--session_name', type=str, default='session', metavar='N',
                        help='Session name')
    parser.add_argument('--previous_session', type=str, default='session', metavar='N',
                        help='Previous session name')

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # set session name depending on input arg
    config["session_name"] = args.session_name
    config["previous_session"] = args.previous_session

    # dataloader setup
    test_dataset = KittiDataset(config, set='test')
    test_sampler = SequentialWindowBatchSampler(batch_size=config["test_loader"]["batch_size"],
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
    tester = Tester(model, test_loader, config)

    # train
    tester.test()

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)

