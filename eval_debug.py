import os
import argparse
import json
import torch

from datasets.oxford import get_dataloaders
from networks.svd_pose_model import SVDPoseModel
from networks.steam_pose_model import SteamPoseModel
from utils.utils import supervised_loss, pointmatch_loss, get_lr
from utils.monitor import SteamEvalMonitor
from datasets.transforms import augmentBatch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/steam_eval.json', type=str, help='eval config file path')
    args = parser.parse_args()

    # load config
    with open(args.config) as f:
        eval_config = json.load(f)
    with open(eval_config["train_config"]) as f:
        train_config = json.load(f)
    train_config['gpuid'] = eval_config['gpuid']

    train_loader, valid_loader, _ = get_dataloaders(train_config)

    model = SteamPoseModel(train_config).to(eval_config['gpuid'])

    model.load_state_dict(torch.load(eval_config['model_path']), strict=False)
    model.eval()

    monitor = SteamEvalMonitor(model, valid_loader, train_config, eval_config)
    monitor.evaluate()
