import io
import torch
import torchvision.utils as vutils
from torchvision.transforms import ToTensor
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import get_transform2, enforce_orthog, get_inverse_tf

def convert_plt_to_img():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return PIL.Image.open(buf)

def convert_plt_to_tensor():
    return ToTensor()(convert_plt_to_img())

def draw_batch(batch, out, config):
    # Draw radar image
    radar = batch['input'][0].squeeze().numpy()
    fig, ax = plt.subplots()
    plt.imshow(radar, cmap='gray')
    radar_img = convert_plt_to_tensor()

    # Draw keypoint matches
    src = out['src'][0].squeeze().detach().cpu().numpy()
    tgt = out['tgt'][0].squeeze().detach().cpu().numpy()
    match_weights = out['match_weights'][0].squeeze().detach().cpu().numpy()

    nms = config['vis_keypoint_nms']
    max_w = np.max(match_weights)
    plt.imshow(radar, cmap='gray')
    for i in range(src.shape[0]):
        if match_weights[i] < nms * max_w:
            continue
        plt.plot([src[i, 0], tgt[i, 0]], [src[i, 1], tgt[i, 1]], c='w', linewidth=2, zorder=2)
        plt.scatter(src[i, 0], src[i, 1], c='g', s=5, zorder=3)
        plt.scatter(tgt[i, 0], tgt[i, 1], c='r', s=5, zorder=4)
    match_img = convert_plt_to_tensor()

    # Draw scores
    scores = out['scores'][0].squeeze().detach().cpu().numpy()
    fig, ax = plt.subplots()
    plt.imshow(scores, cmap='inferno')
    score_img = convert_plt_to_tensor()

    return vutils.make_grid([radar_img, score_img, match_img])

def plot_sequences(T_gt, R_pred, t_pred, seq_len, returnTensor=True):
    seq_indices = []
    idx = 0
    for s in seq_len:
        seq_indices.append(list(range(idx, idx + s - 1)))
        idx += (s - 1)

    imgs = []
    for indices in seq_indices:
        T_gt_ = np.identity(4)
        T_pred_ = np.identity(4)
        x_gt = []
        y_gt = []
        x_pred = []
        y_pred = []
        for i in indices:
            T_gt_ = np.matmul(T_gt[i], T_gt_)
            T_pred_ = np.matmul(get_transform2(R_pred[i], t_pred[i]), T_pred_)
            enforce_orthog(T_gt_)
            enforce_orthog(T_pred_)
            T_gt_temp = get_inverse_tf(T_gt_)
            T_pred_temp = get_inverse_tf(T_pred_)
            x_gt.append(T_gt_temp[0, 3])
            y_gt.append(T_gt_temp[1, 3])
            x_pred.append(T_pred_temp[0, 3])
            y_pred.append(T_pred_temp[1, 3])

        fig, ax = plt.subplots()
        plt.grid(which='both', linestyle='--', alpha=0.5)
        plt.axes().set_aspect('equal')
        plt.plot(x_gt, y_gt, 'k', label='GT')
        plt.plot(x_pred, y_pred, 'r', label='PRED')
        if returnTensor:
            imgs.append(convert_plt_to_tensor())
        else:
            imgs.append(convert_plt_to_img())
    return imgs
