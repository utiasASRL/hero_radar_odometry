import io
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from torchvision.transforms import ToTensor
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
    """Creates an image of the radar scan, scores, and keypoint matches for a single batch."""
    # Draw radar image
    radar = batch['data'][0].squeeze().numpy()
    plt.subplots()
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
    plt.subplots()
    plt.imshow(scores, cmap='inferno')
    score_img = convert_plt_to_tensor()

    return vutils.make_grid([radar_img, score_img, match_img])

def draw_batch_steam(batch, out, config):
    """Creates an image of the radar scan, scores, and keypoint matches for a single batch."""
    # Draw radar image
    radar = batch['data'][0].squeeze().numpy()
    radar_tgt = batch['data'][-1].squeeze().numpy()
    plt.imshow(np.concatenate((radar, radar_tgt), axis=1), cmap='gray')
    plt.title('radar src-tgt pair')
    radar_img = convert_plt_to_tensor()

    # Draw keypoint matches
    src = out['src_rc'][-1].squeeze().detach().cpu().numpy()
    tgt = out['tgt_rc'][-1].squeeze().detach().cpu().numpy()
    match_weights = np.exp(out['match_weights'][-1].squeeze().detach().cpu().numpy())
    keypoint_ints = out['keypoint_ints']

    ids = torch.nonzero(keypoint_ints[-1, 0] > 0, as_tuple=False).squeeze(1)
    ids_cpu = ids.cpu()

    nms = config['vis_keypoint_nms']    # inverse variance
    # max_w = np.max(match_weights)
    plt.imshow(np.concatenate((radar, radar_tgt), axis=1), cmap='gray')
    delta = radar.shape[1]
    for i in range(src.shape[0]):
        if match_weights[i] < nms:
            continue
        if i in ids_cpu:
            custom_colour = 'g'
            plt.plot([src[i, 0], tgt[i, 0] + delta], [src[i, 1], tgt[i, 1]], c='y', linewidth=0.5, zorder=2)
            plt.scatter(src[i, 0], src[i, 1], c=custom_colour, s=5, zorder=3)
            plt.scatter(tgt[i, 0] + delta, tgt[i, 1], c=custom_colour, s=5, zorder=4)
    plt.title('matches')
    match_img = convert_plt_to_tensor()

    plt.imshow(np.concatenate((radar, radar_tgt), axis=0), cmap='gray')
    delta = radar.shape[1]
    for i in range(src.shape[0]):
        if match_weights[i] < nms:
            continue
        if i in ids_cpu:
            custom_colour = 'g'
            plt.plot([src[i, 0], tgt[i, 0]], [src[i, 1], tgt[i, 1] + delta], c='y', linewidth=0.5, zorder=2)
            plt.scatter(src[i, 0], src[i, 1], c=custom_colour, s=5, zorder=3)
            plt.scatter(tgt[i, 0], tgt[i, 1] + delta, c=custom_colour, s=5, zorder=4)
    plt.title('matches')
    match_img2 = convert_plt_to_tensor()

    # Draw scores
    scores = out['scores'][-1].squeeze().detach().cpu().numpy()
    plt.imshow(scores, cmap='inferno')
    plt.colorbar()
    plt.title('log inverse variance (weight score)')
    score_img = convert_plt_to_tensor()

    # Draw detector scores
    detector_scores = out['detector_scores'][-1].squeeze().detach().cpu().numpy()
    plt.imshow(detector_scores, cmap='inferno')
    plt.colorbar()
    plt.title('detector score')
    dscore_img = convert_plt_to_tensor()

    # Draw point-to-point error
    src_p = out['src'][-1].squeeze().T
    tgt_p = out['tgt'][-1].squeeze().T
    R_tgt_src = out['R'][0, -1, :2, :2]
    t_st_in_t = out['t'][0, -1, :2, :]
    error = tgt_p - (R_tgt_src @ src_p + t_st_in_t)
    mah = torch.sqrt(torch.sum(error * error * torch.exp(out['match_weights'][-1]), dim=0).squeeze())
    error2_sqrt = torch.sqrt(torch.sum(error * error, dim=0).squeeze())

    plt.imshow(radar, cmap='gray')
    plt.scatter(src[ids_cpu, 0], src[ids_cpu, 1], c=error2_sqrt[ids_cpu].detach().cpu().numpy(), s=5, zorder=2, cmap='rainbow')
    plt.colorbar()
    plt.title('P2P error')
    p2p_img = convert_plt_to_tensor()

    plt.imshow(radar, cmap='gray')
    plt.scatter(src[ids_cpu, 0], src[ids_cpu, 1], c=mah[ids_cpu].detach().cpu().numpy(), s=5, zorder=2, cmap='rainbow')
    plt.colorbar()
    plt.title('MAH')
    mah_img = convert_plt_to_tensor()

    return vutils.make_grid([dscore_img, score_img, radar_img]), vutils.make_grid([match_img, match_img2]), \
           vutils.make_grid([p2p_img, mah_img])

def plot_sequences(T_gt, R_pred, t_pred, seq_len, returnTensor=True):
    """Creates a top-down plot of the predicted odometry results vs. ground truth."""
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

        plt.subplots()
        plt.grid(which='both', linestyle='--', alpha=0.5)
        plt.axes().set_aspect('equal')
        plt.plot(x_gt, y_gt, 'k', label='GT')
        plt.plot(x_pred, y_pred, 'r', label='PRED')
        if returnTensor:
            imgs.append(convert_plt_to_tensor())
        else:
            imgs.append(convert_plt_to_img())
    return imgs
