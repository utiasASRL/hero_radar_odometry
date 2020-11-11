import os
import argparse
import json

from utils.trainer import Trainer
from datasets.oxford import *
from networks.svd_pose import SVDPoseModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/radar.json', type=str, help='config file path')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    train_loader, valid_loader = get_dataloader(config)

    model = SVDPoseModel(config)

    trainer = Trainer(model, train_loader, valid_loader, config)
    trainer.train()
