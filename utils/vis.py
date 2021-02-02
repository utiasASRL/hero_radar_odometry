import io
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from torchvision.transforms import ToTensor
from utils.utils import get_transform2, enforce_orthog, get_inverse_tf
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

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
    # match_weights = np.exp(out['match_weights'][-1].squeeze().detach().cpu().numpy())
    keypoint_ints = out['keypoint_ints']

    ids = torch.nonzero(keypoint_ints[-1, 0] > 0, as_tuple=False).squeeze(1)
    ids_cpu = ids.cpu()

    # nms = config['vis_keypoint_nms']    # inverse variance
    # max_w = np.max(match_weights)
    plt.imshow(np.concatenate((radar, radar_tgt), axis=1), cmap='gray')
    delta = radar.shape[1]
    for i in range(src.shape[0]):
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
        if i in ids_cpu:
            custom_colour = 'g'
            plt.plot([src[i, 0], tgt[i, 0]], [src[i, 1], tgt[i, 1] + delta], c='y', linewidth=0.5, zorder=2)
            plt.scatter(src[i, 0], src[i, 1], c=custom_colour, s=5, zorder=3)
            plt.scatter(tgt[i, 0], tgt[i, 1] + delta, c=custom_colour, s=5, zorder=4)
    plt.title('matches')
    match_img2 = convert_plt_to_tensor()

    # Draw scores
    scores = out['scores'][-1]
    if scores.size(0) == 3:
        scores = scores[1] + scores[2]
    scores = scores.squeeze().detach().cpu().numpy()
    plt.imshow(scores, cmap='inferno')
    plt.colorbar()
    plt.title('log det weight (weight score vis)')
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
    # mah = torch.sqrt(torch.sum(error * error * torch.exp(out['match_weights'][-1]), dim=0).squeeze())
    error2_sqrt = torch.sqrt(torch.sum(error * error, dim=0).squeeze())

    plt.imshow(radar, cmap='gray')
    plt.scatter(src[ids_cpu, 0], src[ids_cpu, 1], c=error2_sqrt[ids_cpu].detach().cpu().numpy(), s=5, zorder=2, cmap='rainbow')
    plt.clim(0.0, 1.0)
    plt.colorbar()
    plt.title('P2P error')
    p2p_img = convert_plt_to_tensor()

    # plt.imshow(radar, cmap='gray')
    # plt.scatter(src[ids_cpu, 0], src[ids_cpu, 1], c=mah[ids_cpu].detach().cpu().numpy(), s=5, zorder=2, cmap='rainbow')
    # plt.colorbar()
    # plt.title('MAH')
    # mah_img = convert_plt_to_tensor()

    return vutils.make_grid([dscore_img, score_img, radar_img]), vutils.make_grid([match_img, match_img2]), \
           vutils.make_grid([p2p_img])

def draw_batch_steam_eval(batch, out, model):
    # get radar image
    radar = batch['data'][0].squeeze().numpy()
    radar_tgt = batch['data'][-1].squeeze().numpy()

    # Draw keypoint matches
    src = out['src_rc'][-1].squeeze().detach().cpu().numpy()
    tgt = out['tgt_rc'][-1].squeeze().detach().cpu().numpy()
    keypoint_ints = out['keypoint_ints']

    ids = torch.nonzero(keypoint_ints[-1, 0] > 0, as_tuple=False).squeeze(1)
    ids_cpu = ids.cpu()

    plt.imshow(np.concatenate((radar, radar_tgt), axis=1), cmap='gray')
    delta = radar.shape[1]
    for i in range(src.shape[0]):
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
        if i in ids_cpu:
            custom_colour = 'g'
            plt.plot([src[i, 0], tgt[i, 0]], [src[i, 1], tgt[i, 1] + delta], c='y', linewidth=0.5, zorder=2)
            plt.scatter(src[i, 0], src[i, 1], c=custom_colour, s=5, zorder=3)
            plt.scatter(tgt[i, 0], tgt[i, 1] + delta, c=custom_colour, s=5, zorder=4)
    plt.title('matches')
    match_img2 = convert_plt_to_tensor()

    # draw radar image
    plt.imshow(np.concatenate((radar, radar_tgt), axis=1), cmap='gray')
    delta = radar.shape[1]
    for i in range(src.shape[0]):
        if i in ids_cpu:
            custom_colour = 'g'
            plt.scatter(src[i, 0], src[i, 1], c=custom_colour, s=5, zorder=3)
            plt.scatter(tgt[i, 0] + delta, tgt[i, 1], c=custom_colour, s=5, zorder=4)
    plt.title('radar src-tgt pair')
    radar_img = convert_plt_to_tensor()

    # plot points and covariance
    _, ax = plt.subplots()
    plt.axis('equal')
    tgt_coords = out['tgt'][-1].squeeze().detach().cpu().numpy()
    match_weights = out['match_weights']
    weights_mat, weights_d = model.solver.convert_to_weight_matrix(match_weights[-1].T, match_weights.size(0)-1)
    for i in range(src.shape[0]):
        if i in ids_cpu:
            custom_colour = 'g'
            cov = np.linalg.inv(weights_mat[i, :2, :2].detach().cpu().numpy())
            confidence_ellipse_2D(tgt_coords[i, :2], cov, ax, n_std=3.0, facecolor='y')
            plt.scatter(tgt_coords[i, 0], tgt_coords[i, 1], c=custom_colour, s=5, zorder=4)
    points_cov_img = convert_plt_to_tensor()

    return vutils.make_grid([radar_img, points_cov_img]), vutils.make_grid([match_img, match_img2])


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

def confidence_ellipse_2D(mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Modified for 2x1 mean and 2x2 covariance input  (David Yoon)
    -----
    Create a plot of the covariance confidence ellipse of `x` and `y`

    See how and why this works: https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html

    This function has made it into the matplotlib examples collection:
    https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py

    Or, once matplotlib 3.1 has been released:
    https://matplotlib.org/gallery/index.html#statistics

    I update this gist according to the version there, because thanks to the matplotlib community
    the code has improved quite a bit.
    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    Returns
    -------
    matplotlib.patches.Ellipse
    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)