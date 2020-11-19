import argparse
import json
import torch

from datasets.oxford import get_dataloaders
from networks.svd_pose_model import SVDPoseModel
from utils.utils import supervised_loss
from utils.monitor import Monitor
from datasets.transforms import augmentBatch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/radar.json', type=str, help='config file path')
    parser.add_argument('--pretrain', default=None, type=str, help='pretrain checkpoint path')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    train_loader, valid_loader, _ = get_dataloaders(config)

    model = SVDPoseModel(config).to(config['gpuid'])

    if args.pretrain is not None:
        model.load_state_dict(torch.load(args.pretrain), strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    monitor = Monitor(model, valid_loader, config)

    model.train()

    for batchi, batch in enumerate(train_loader):
        if config['augmentation']['augment']:
            batch = augmentBatch(batch, config)
        optimizer.zero_grad()
        out = model(batch)
        loss, R_loss, t_loss = supervised_loss(out['R'], out['t'], batch, config)
        if loss.requires_grad:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_norm'])
        optimizer.step()
        monitor.step(batchi, loss, R_loss, t_loss, batch, out)
        if batchi >= config['max_iterations']:
            break
