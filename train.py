import os
import argparse
import json
import torch
import numpy as np

from datasets.oxford import get_dataloaders
from datasets.boreas import get_dataloaders_boreas
from networks.svd_pose_model import SVDPoseModel
from networks.steam_pose_model import SteamPoseModel
from utils.utils import supervised_loss, pointmatch_loss, get_lr
from utils.monitor import SVDMonitor, SteamMonitor
from datasets.transforms import augmentBatch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
#torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(8)
torch.multiprocessing.set_sharing_strategy('file_system')
print(torch.__version__)
print(torch.version.cuda)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/steam.json', type=str, help='config file path')
    parser.add_argument('--pretrain', default=None, type=str, help='pretrain checkpoint path')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    if config['dataset'] == 'oxford':
        train_loader, valid_loader, _ = get_dataloaders(config)
    elif config['dataset'] == 'boreas':
        train_loader, valid_loader, _ = get_dataloaders_boreas(config)

    if config['model'] == 'SVDPoseModel':
        model = SVDPoseModel(config).to(config['gpuid'])
    elif config['model'] == 'SteamPoseModel':
        model = SteamPoseModel(config).to(config['gpuid'])

    ckpt_path = None
    if os.path.isfile(config['log_dir'] + 'latest.pt'):
        ckpt_path = config['log_dir'] + 'latest.pt'
    elif args.pretrain is not None:
        ckpt_path = args.pretrain

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2.5e4 / config['val_rate'], factor=0.5)
    if config['model'] == 'SVDPoseModel':
        monitor = SVDMonitor(model, valid_loader, config)
    elif config['model'] == 'SteamPoseModel':
        monitor = SteamMonitor(model, valid_loader, config)
    start_epoch = 0

    if ckpt_path is not None:
        try:
            print('Loading from checkpoint: ' + ckpt_path)
            checkpoint = torch.load(ckpt_path, map_location=torch.device(config['gpuid']))
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            monitor.counter = checkpoint['counter']
            print('success')
        except Exception as e:
            print(e)
            print('Defaulting to legacy checkpoint style')
            model.load_state_dict(checkpoint, strict=False)
            print('success')
    #model = torch.nn.DataParallel(model)
    if not os.path.isfile(config['log_dir'] + args.config):
        os.system('cp ' + args.config + ' ' + config['log_dir'])

    model.train()

    # check for network freeze
    if 'freeze_network' in config['steam']:
        if config['steam']['freeze_network']:
            print('Freezing network parameters...')
            for param in model.parameters():
                param.requires_grad = False
            model.q.requires_grad = True

    for epoch in range(start_epoch, config['max_epochs']):
        for batchi, batch in enumerate(train_loader):
            if config['augmentation']['augment']:
                batch = augmentBatch(batch, config)
            optimizer.zero_grad()
            try:
                out = model(batch)
            except RuntimeError as e:
                print(e)
                print('WARNING: exception encountered... skipping this batch.')
                continue
            if config['model'] == 'SVDPoseModel':
                if config['loss'] == 'supervised_loss':
                    loss, dict_loss = supervised_loss(out['R'], out['t'], batch, config)
                elif config['loss'] == 'pointmatch_loss':
                    loss, dict_loss = pointmatch_loss(out, batch, config)
            elif config['model'] == 'SteamPoseModel':
                loss, dict_loss = model.loss(out['src'], out['tgt'], out['match_weights'], out['keypoint_ints'], out['scores'], batch)
            if loss == 0:
                print("No movement predicted. Skipping mini-batch.")
                continue
            if loss.requires_grad:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_norm'])
            optimizer.step()
            if (monitor.counter + 1) % config['save_rate'] == 0:
                with torch.no_grad():
                    model.eval()
                    mname = os.path.join(config['log_dir'], '{}.pt'.format(monitor.counter + 1))
                    print('saving model', mname)
                    torch.save({
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'counter': monitor.counter,
                                'epoch': epoch,
                                }, mname)
                    model.train()
            if (monitor.counter + 1) % config['backup_rate'] == 0:
                with torch.no_grad():
                    model.eval()
                    mname = os.path.join(config['log_dir'], 'latest.pt')
                    print('saving model', mname)
                    torch.save({
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'counter': monitor.counter,
                                'epoch': epoch,
                                }, mname)
                    model.train()

            valid_metric = monitor.step(loss, dict_loss)
            if valid_metric is not None:
                scheduler.step(valid_metric)
                monitor.writer.add_scalar('val/learning_rate', get_lr(optimizer), monitor.counter)
            if monitor.counter >= config['max_iterations']:
                break
