import argparse
import json
from time import time
import numpy as np
import torch
import pickle

from datasets.oxford import get_dataloaders
from datasets.boreas import get_dataloaders_boreas
from networks.under_the_radar import UnderTheRadar
from networks.hero import HERO
from utils.utils import computeMedianError, computeKittiMetrics, saveKittiErrors, save_in_yeti_format, get_T_ba
from utils.utils import load_icra21_results, getStats, get_inverse_tf
from utils.vis import plot_sequences

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True

def get_folder_from_file_path(path):
    elems = path.split('/')
    newpath = ""
    for j in range(0, len(elems) - 1):
        newpath += elems[j] + "/"
    return newpath

if __name__ == '__main__':
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/steam.json', type=str, help='config file path')
    parser.add_argument('--pretrain', default=None, type=str, help='pretrain checkpoint path')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    root = get_folder_from_file_path(args.pretrain)
    seq_nums = config['test_split']
    if config['model'] == 'UnderTheRadar':
        model = UnderTheRadar(config).to(config['gpuid'])
    elif config['model'] == 'HERO':
        model = HERO(config).to(config['gpuid'])
        model.solver.sliding_flag = True
    assert(args.pretrain is not None)
    checkpoint = torch.load(args.pretrain, map_location=torch.device(config['gpuid']))
    failed = False
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    except Exception as e:
        print(e)
        failed = True
    if failed:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()

    T_gt_ = []
    T_pred_ = []
    t_errs = []
    r_errs = []
    time_used_ = []

    for seq_num in seq_nums:
        time_used = []
        T_gt = []
        T_pred = []
        timestamps = []
        config['test_split'] = [seq_num]
        if config['dataset'] == 'oxford':
            _, _, test_loader = get_dataloaders(config)
        elif config['dataset'] == 'boreas':
            _, _, test_loader = get_dataloaders_boreas(config)
        seq_lens = test_loader.dataset.seq_lens
        print(seq_lens)
        seq_names = test_loader.dataset.sequences
        print('Evaluating sequence: {} : {}'.format(seq_num, seq_names[0]))
        nkpts = []
        for batchi, batch in enumerate(test_loader):
            ts = time()
            if (batchi + 1) % config['print_rate'] == 0:
                print('Eval Batch {} / {}: {:.2}s'.format(batchi, len(test_loader), np.mean(time_used[-config['print_rate']:])))
            with torch.no_grad():
                try:
                    out = model(batch)
                    y = torch.nonzero(out['keypoint_ints'][0] > 0.5, as_tuple=False).squeeze()
                    nkpts.append(len(y))
                except RuntimeError as e:
                    print(e)
                    continue
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
                
                #Tp = get_T_ba(out, a=w, b=w+1)
                #Tpinv = get_inverse_tf(Tp)
                #T_pred.append(Tp)


                #Tp = get_T_ba(out, a=w, b=w+1)
                #Tpinv = get_inverse_tf(Tp)
                #Tpinv[1, 3] *= -1
                #Tp = get_inverse_tf(Tpinv)
                #T_pred.append(Tp)

                T_pred.append(get_T_ba(out, a=w, b=w+1))
                print(batchi)
                print('T_gt:\n{}'.format(T_gt[-1]))
                print('T_pred:\n{}'.format(T_pred[-1]))
                timestamps.append(batch['times'][w].numpy().squeeze())
            time_used.append(time() - ts)
        T_gt_.extend(T_gt)
        T_pred_.extend(T_pred)
        time_used_.extend(time_used)
        t_err, r_err, err = computeKittiMetrics(T_gt, T_pred, [len(T_gt)])
        print('SEQ: {} : {}'.format(seq_num, seq_names[0]))
        print('KITTI t_err: {} %'.format(t_err))
        print('KITTI r_err: {} deg/m'.format(r_err))
        t_errs.append(t_err)
        r_errs.append(r_err)
        save_in_yeti_format(T_gt, T_pred, timestamps, [len(T_gt)], seq_names, root)
        pickle.dump([T_gt, T_pred, timestamps], open(root + 'odom' + seq_names[0] + '.obj', 'wb'))
        T_icra = None
        if config['dataset'] == 'oxford':
            T_icra = load_icra21_results('./results/icra21/', seq_names, seq_lens)
        fname = root + seq_names[0] + '.pdf'
        plot_sequences(T_gt, T_pred, [len(T_gt)], returnTensor=False, T_icra=T_icra, savePDF=True, fnames=[fname])

    print('time_used: {}'.format(sum(time_used_) / len(time_used_)))
    results = computeMedianError(T_gt_, T_pred_)
    with open('errs.obj', 'wb') as f:
            pickle.dump([nkpts, results[-2], results[-1]], f)
    print('dt: {} sigma_dt: {} dr: {} sigma_dr: {}'.format(results[0], results[1], results[2], results[3]))

    t_err = np.mean(t_errs)
    r_err = np.mean(r_errs)
    print('Average KITTI metrics over all test sequences:')
    print('KITTI t_err: {} %'.format(t_err))
    print('KITTI r_err: {} deg/m'.format(r_err))
