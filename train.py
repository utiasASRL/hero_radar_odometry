import os
import argparse
import json

from utils.trainer import Trainer
from datasets.oxford import *
from networks.svd_pose import SVDPoseModel
from utils.monitor import Monitor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/radar.json', type=str, help='config file path')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    train_loader, valid_loader = get_dataloader(config)

    model = SVDPoseModel(config).to(config['gpuid'])

    if pretrain_path is not None:
        model.load_state_dict(torch.load(pretrain_path), strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    monitor = Monitor(model, log_dir, valid_loader, config)

    model.train()

    for batchi, batch in enumerate(train_loader):
        optimizer.zero_grad()
        R_tgt_src_pred, t_tgt_src_pred = model(batch)
        loss, R_loss, t_loss = supervised_loss(R_tgt_src_pred, t_tgt_src_pred, batch)
        if loss.requires_grad:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_norm'])
        optimizer.step()
        monitor.step(batchi, loss, R_loss, t_loss, batch, R_tgt_src_pred, t_tgt_src_pred)
        if batchi >= config['max_iterations']:
            break
