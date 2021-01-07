import argparse
import json
import torch

from datasets.oxford import get_dataloaders
from networks.svd_pose_model import SVDPoseModel
from networks.steam_pose_model import SteamPoseModel
from utils.utils import supervised_loss
from utils.monitor import SVDMonitor, SteamMonitor
from datasets.transforms import augmentBatch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/radar.json', type=str, help='config file path')
    parser.add_argument('--pretrain', default=None, type=str, help='pretrain checkpoint path')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    train_loader, valid_loader, _ = get_dataloaders(config)

    if config['model'] == 'SVDPoseModel':
        model = SVDPoseModel(config).to(config['gpuid'])
    elif config['model'] == 'SteamPoseModel':
        model = SteamPoseModel(config).to(config['gpuid'])

    if args.pretrain is not None:
        model.load_state_dict(torch.load(args.pretrain), strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    if config['model'] == 'SVDPoseModel':
        monitor = SVDMonitor(model, valid_loader, config)
    elif config['model'] == 'SteamPoseModel':
        monitor = SteamMonitor(model, valid_loader, config)

    model.train()

    step = 0
    for epoch in range(config['max_epochs']):
        for batchi, batch in enumerate(train_loader):
            if config['augmentation']['augment']:
                batch = augmentBatch(batch, config)
            optimizer.zero_grad()
            out = model(batch)
            if config['model'] == 'SVDPoseModel':
                if config['loss'] == 'supervised_loss':
                    loss, dict_loss = supervised_loss(out['R'], out['t'], batch, config)
                elif config['loss'] == 'pointmatch_loss':
                    loss, dict_loss = pointmatch_loss(out['R'], out['t'], out['tgt'], out['src'], config)
            elif config['model'] == 'SteamPoseModel':
                loss, dict_loss = model.loss(out['src'], out['tgt'], out['match_weights'], out['keypoint_ints'])
            if loss == 0:
                print("No movement predicted. Skipping mini-batch.")
                continue
            if loss.requires_grad:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_norm'])
            optimizer.step()
            step = batchi + epoch * len(train_loader.dataset)
            monitor.step(step, loss, dict_loss)
            if step >= config['max_iterations']:
                break
