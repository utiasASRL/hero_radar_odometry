import argparse
import json
from time import time
import numpy as np
import torch
import pickle

from datasets.oxford import get_dataloaders
from networks.svd_pose_model import SVDPoseModel
from networks.steam_pose_model import SteamPoseModel
from utils.utils import computeMedianError, computeKittiMetrics, saveKittiErrors, save_in_yeti_format, get_T_ba
from utils.utils import load_icra21_results, getStats
from utils.vis import plot_sequences

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

print(torch.__version__)
print(torch.version.cuda)
torch.multiprocessing.set_sharing_strategy('file_system')

def get_folder_from_file_path(path):
    elems = path.split('/')
    newpath = ""
    for j in range(0, len(elems) - 1):
        newpath += elems[j] + "/"
    return newpath

if __name__ == '__main__':
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/eval.json', type=str, help='config file path')
    parser.add_argument('--pretrain', default=None, type=str, help='pretrain checkpoint path')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    config['gpuid'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = get_folder_from_file_path(args.pretrain)
    seq_nums = config['test_split']
    if config['model'] == 'SVDPoseModel':
        model = SVDPoseModel(config)
    elif config['model'] == 'SteamPoseModel':
        model = SteamPoseModel(config)
        model.solver.sliding_flag = True
        # self.model.solver.log_det_thres_flag = True
    model.to(config['gpuid'])
    assert(args.pretrain is not None)
    model.load_state_dict(torch.load(args.pretrain, map_location=torch.device(config['gpuid'])), strict=False)
    model.eval()

    T_gt_ = []
    T_pred_ = []
    err_ = []
    time_used_ = []

    for seq_num in seq_nums:
        time_used = []
        T_gt = []
        T_pred = []
        timestamps = []
        config['test_split'] = [seq_num]
        _, _, test_loader = get_dataloaders(config)
        seq_lens = test_loader.dataset.seq_lens
        print(seq_lens)
        seq_names = test_loader.dataset.sequences
        print('Evaluating sequence: {} : {}'.format(seq_num, seq_names[0]))
        for batchi, batch in enumerate(test_loader):
            ts = time()
            if (batchi + 1) % config['print_rate'] == 0:
                print('Eval Batch {} / {}: {:.2}s'.format(batchi, len(test_loader), np.mean(time_used[-config['print_rate']:])))
            with torch.no_grad():
                out = model(batch)
            if batchi == len(test_loader) - 1:
                # append entire window
                for w in range(batch['T_21'].size(0)-1):
                    T_gt.append(batch['T_21'][w].numpy().squeeze())
                    T_pred.append(get_T_ba(out, a=w, b=w+1))
                    timestamps.append(batch['times'][w].numpy().squeeze())
            else:
                # append only the back of window
                w = 0
                T_gt.append(batch['T_21'][w].numpy().squeeze())
                T_pred.append(get_T_ba(out, a=w, b=w+1))
                timestamps.append(batch['times'][w].numpy().squeeze())
            time_used.append(time() - ts)
        T_gt_.extend(T_gt)
        T_pred_.extend(T_pred)
        time_used_.extend(time_used)
        t_err, r_err, err = computeKittiMetrics(T_gt, T_pred, seq_lens)
        print('SEQ: {} : {}'.format(seq_num, seq_names[0]))
        print('KITTI t_err: {} %'.format(t_err))
        print('KITTI r_err: {} deg/m'.format(r_err))
        err_.extend(err)
        save_in_yeti_format(T_gt, T_pred, timestamps, seq_lens, seq_names, root)
        pickle.dump([T_gt, T_pred, timestamps], open(root + 'odom' + seq_names[0] + '.obj', 'wb'))
        T_icra = load_icra21_results('/h/keenan/RadarLocNet/results/icra21/', seq_names, seq_lens)
        fname = root + seq_names[0] + '.pdf'
        plot_sequences(T_gt, T_pred, seq_lens, returnTensor=False, T_icra=T_icra, savePDF=True, fnames=[fname])

    print('time_used: {}'.format(sum(time_used_) / len(time_used_)))
    results = computeMedianError(T_gt_, T_pred_)
    print('dt: {} sigma_dt: {} dr: {} sigma_dr: {}'.format(results[0], results[1], results[2], results[3]))

    t_err, r_err = getStats(err_)
    print('Average KITTI metrics over all test sequences:')
    print('KITTI t_err: {} %'.format(t_err))
    print('KITTI r_err: {} deg/m'.format(r_err))
    saveKittiErrors(err_, root + "kitti_err.obj")

