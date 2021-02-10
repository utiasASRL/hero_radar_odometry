import argparse
import json
from time import time
import numpy as np
import torch

from datasets.oxford import get_dataloaders
from networks.svd_pose_model import SVDPoseModel
from networks.steam_pose_model import SteamPoseModel
from utils.utils import computeMedianError, computeKittiMetrics, saveKittiErrors, save_in_yeti_format, get_T_ba
from utils.utils import load_icra21_results
from utils.vis import plot_sequences

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def get_folder_from_file_path(path):
    elems = path.split('/')
    newpath = ""
    for j in range(0, len(elems) - 1):
        newpath += elems[j] + "/"
    return newpath

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/eval.json', type=str, help='config file path')
    parser.add_argument('--pretrain', default=None, type=str, help='pretrain checkpoint path')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    _, _, test_loader = get_dataloaders(config)
    seq_lens = test_loader.dataset.seq_lens
    seq_names = test_loader.dataset.sequences

    if config['model'] == 'SVDPoseModel':
        model = SVDPoseModel(config).to(config['gpuid'])
    elif config['model'] == 'SteamPoseModel':
        model = SteamPoseModel(config).to(config['gpuid'])
        model.solver.sliding_flag = True
        model.log_det_thres_flag = True
    assert(args.pretrain is not None)
    model.load_state_dict(torch.load(args.pretrain, map_location=torch.device(config['gpuid'])), strict=False)
    model.eval()

    time_used = []
    T_gt = []
    T_pred = []
    R_pred = []
    t_pred = []
    timestamps = []
    for batchi, batch in enumerate(test_loader):
        ts = time()
        if (batchi + 1) % config['print_rate'] == 0:
            print('Eval Batch {}: {:.2}s'.format(batchi, np.mean(time_used[-config['print_rate']:])))
        with torch.no_grad():
            out = model(batch)
        if batchi == len(test_loader):
            # append entire window
            for w in range(batch['T_21'].size(0)-1):
                T_gt.append(batch['T_21'][w].numpy().squeeze())
                T_pred.append(get_T_ba(out, b=w+1, a=w))
                timestamps.append(batch['times'][w].numpy().squeeze())
        else:
            # append only the back of window
            w = 0
            T_gt.append(batch['T_21'][w].numpy().squeeze())
            T_pred.append(get_T_ba(out, b=w+1, a=w))
            timestamps.append(batch['times'][w].numpy().squeeze())
        time_used.append(time() - ts)

    print('time_used: {}'.format(sum(time_used) / len(time_used)))
    results = computeMedianError(T_gt, T_pred)
    print('dt: {} sigma_dt: {} dr: {} sigma_dr: {}'.format(results[0], results[1], results[2], results[3]))

    t_err, r_err, err = computeKittiMetrics(T_gt, T_pred, seq_lens)
    print('KITTI t_err: {} %'.format(t_err))
    print('KITTI r_err: {} deg/m'.format(r_err))
    root = get_folder_from_file_path(args.pretrain)
    saveKittiErrors(err, root + "kitti_err.obj")

    save_in_yeti_format(T_gt, T_pred, timestamps, seq_lens, seq_names, root)

    T_icra = load_icra21_results('./results/icra21/', seq_names, seq_lens)
    imgs = plot_sequences(T_gt, T_pred, seq_lens, returnTensor=False, T_icra=T_icra)
    for i, img in enumerate(imgs):
        imgs[i].save(root + seq_names[i] + '.png')
