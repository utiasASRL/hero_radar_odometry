import argparse
import json
import torch
import os
import numpy as np

from datasets.oxford import get_dataloaders
from networks.svd_pose_model import SVDPoseModel
from networks.steam_pose_model import SteamPoseModel
from utils.utils import supervised_loss
from utils.steam_monitor import SteamMonitor
from datasets.transforms import augmentBatch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/steam.json', type=str, help='config file path')
    parser.add_argument('--pretrain', default=None, type=str, help='pretrain checkpoint path')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    train_loader, valid_loader, _ = get_dataloaders(config)

    model = SteamPoseModel(config).to(config['gpuid'])

    if args.pretrain is not None:
        model.load_state_dict(torch.load(args.pretrain), strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    monitor = SteamMonitor(model, valid_loader, config)
    # temporary logging and saving
    # if not os.path.exists(config['log_dir']):
    #     os.makedirs(config['log_dir'])
    # else:
    #     assert False, "Session name already exists!"
    # log_path = os.path.join(config['log_dir'], 'train.txt')
    # checkpoint_path = os.path.join(config['log_dir'], 'chkp.tar')

    model.train()

    step = 0
    for epoch in range(config['max_epochs']):
        for batchi, batch in enumerate(train_loader):
            if config['augmentation']['augment']:
                batch = augmentBatch(batch, config)
            optimizer.zero_grad()
            out = model(batch)
            loss, dict_loss = model.loss(out['src'], out['tgt'], out['match_weights'])
            if loss.requires_grad:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_norm'])
            optimizer.step()
            step = batchi + epoch * len(train_loader.dataset)
            monitor.step(step, loss, dict_loss)

            # temporary print to log file
            # print(epoch, ',', batchi, ',', float(loss))
            # with open(log_path, "a") as file:
            #     message = '{:d},{:d},{:.6f}\n'
            #     file.write(message.format(epoch, batchi, float(loss)))
            #
            # if np.mod(batchi, 500) == 0:
            #     # save out every subepoch
            #     print('saving at subepoch...')
            #     torch.save({'epoch': epoch,
            #                 'model_state_dict': model.state_dict(),
            #                 'optimizer_state_dict': optimizer.state_dict(),
            #                 'loss': float(loss),
            #                 }, checkpoint_path)

            if step >= config['max_iterations']:
                break
