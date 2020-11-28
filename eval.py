import argparse
import json
from time import time
import numpy as np
import torch

from datasets.oxford import get_dataloaders
from networks.svd_pose_model import SVDPoseModel
from utils.utils import computeMedianError, computeKittiMetrics, saveKittiErrors, save_in_yeti_format
from utils.vis import plot_sequences

def get_folder_from_file_path(path):
    elems = path.split('/')
    newpath = ""
    for j in range(0, len(elems) - 1):
        newpath += elems[j] + "/"
    return newpath

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/radar.json', type=str, help='config file path')
    parser.add_argument('--pretrain', default=None, type=str, help='pretrain checkpoint path')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    _, _, test_loader = get_dataloaders(config)
    seq_len = test_loader.dataset.seq_len
    seq_names = test_loader.dataset.sequences

    model = SVDPoseModel(config)
    model.load_state_dict(torch.load(args.pretrain, map_location=torch.device(config['gpuid'])), strict=False)
    model.to(config['gpuid'])
    model.eval()

    time_used = []
    T_gt = []
    R_pred = []
    t_pred = []
    timestamps = []
    for batchi, batch in enumerate(test_loader):
        ts = time()
        if (batchi + 1) % config['print_rate'] == 0:
            print('Eval Batch {}: {:.2}s'.format(batchi, np.mean(time_used[-config['print_rate']:])))
        out = model(batch)
        T_gt.append(batch['T_21'][0].numpy().squeeze())
        R_pred.append(out['R'][0].detach().cpu().numpy().squeeze())
        t_pred.append(out['t'][0].detach().cpu().numpy().squeeze())
        print(T_gt[-1])
        print(R_pred[-1])
        print(t_pred[-1])
        timestamps.append(batch['times'][0].numpy().squeeze())
        time_used.append(time() - ts)

    print('time_used: {}'.format(sum(time_used) / len(time_used)))
    results = computeMedianError(T_gt, R_pred, t_pred)
    print('dt: {} sigma_dt: {} dr: {} sigma_dr: {}'.format(results[0], results[1], results[2], results[3]))

    t_err, r_err, err = computeKittiMetrics(T_gt, R_pred, t_pred, seq_len)
    print('KITTI t_err: {} %'.format(t_err))
    print('KITTI r_err: {} deg/m'.format(r_err))
    root = get_folder_from_file_path(args.pretrain)
    saveKittiErrors(err, root + "kitti_err.obj")

    imgs = plot_sequences(T_gt, R_pred, t_pred, seq_len, returnTensor=False)
    for i, img in enumerate(imgs):
        imgs[i].save(root + seq_names[i] + '.png')

    save_in_yeti_format(T_gt, R_pred, t_pred, timestamps, seq_len, seq_names, root)
