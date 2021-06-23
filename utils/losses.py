import torch
from utils.utils import get_indices, convert_to_weight_matrix

def supervised_loss(R_tgt_src_pred, t_tgt_src_pred, batch, config, alpha=10.0):
    """This function computes the L1 loss between the predicted and groundtruth translation in addition to
        the rotation loss (R_pred.T * R) - I.
    Args:
        R_tgt_src_pred (torch.tensor): (b,3,3) predicted rotation
        t_tgt_src_pred (torch.tensor): (b,3,1) predicted translation
        batch (dict): input data for the batch
        config (json): parsed config file
    Returns:
        svd_loss (float): supervised loss
        dict_loss (dict): a dictionary containing the separate loss components
    """
    T_21 = batch['T_21'].to(config['gpuid'])
    batch_size = R_tgt_src_pred.size(0)
    # Get ground truth transforms
    kp_inds, _ = get_indices(batch_size, config['window_size'])
    T_tgt_src = T_21[kp_inds]
    R_tgt_src = T_tgt_src[:, :3, :3]
    t_tgt_src = T_tgt_src[:, :3, 3].unsqueeze(-1)
    identity = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(config['gpuid'])
    loss_fn = torch.nn.L1Loss()
    R_loss = loss_fn(torch.matmul(R_tgt_src_pred.transpose(2, 1), R_tgt_src), identity)
    t_loss = loss_fn(t_tgt_src_pred, t_tgt_src)
    svd_loss = t_loss + alpha * R_loss
    dict_loss = {'R_loss': R_loss, 't_loss': t_loss}
    return svd_loss, dict_loss

def unsupervised_loss(out, batch, config, solver):
    """This function uses the reprojection between matched pairs of points as a training signal.
        Transformations aligning pairs of frames are estimated using a non-differentiable estimator.
    Args:
        out (dict): The output of the DNN
        batch (dict): input data for the batch
        config (json): parsed config file
        solver: The steam_solver python wrapper class
    Returns:
        total_loss (float): unsupervised loss
        dict_loss (dict): a dictionary containing the separate loss components
    """
    src_coords = out['src']                 # (b*(w-1),N,2) src keypoint locations in metric
    tgt_coords = out['tgt']                 # (b*(w-1),N,2) tgt keypoint locations in metric
    match_weights = out['match_weights']    # (b*(w-1),S,N) match weights S=1=scalar, S=3=matrix
    keypoint_ints = out['keypoint_ints']    # (b*(w-1),1,N) 0==reject, 1==keep
    BW = keypoint_ints.size(0)
    batch_size = int(BW / (self.window_size - 1))
    window_size = config['window_size']
    gpuid = config['gpuid']
    mah_thres = config['steam']['mah_thres']
    expect_approx_opt = config['steam']['expect_approx_opt']
    topk_backup = config['steam']['topk_backup']
    T_aug = []
    if 'T_aug' in batch:
        T_aug = batch['T_aug']
    point_loss = 0
    logdet_loss = 0
    unweighted_point_loss = 0
    zeropad = torch.nn.ZeroPad2d((0, 1, 0, 0))

    # loop through each batch
    bcount = 0
    for b in range(batch_size):
        bcount += 1
        i = b * (window_size-1)    # first index of window
        # loop for each window frame
        for w in range(i, i + window_size - 1):
            # filter by zero intensity patches
            ids = torch.nonzero(keypoint_ints[w, 0] > 0, as_tuple=False).squeeze(1)
            if ids.size(0) == 0:
                print('WARNING: filtering by zero intensity patches resulted in zero keypoints!')
                continue

            # points must be list of N x 3
            points1 = zeropad(src_coords[w, ids]).unsqueeze(-1)    # N x 3 x 1
            points2 = zeropad(tgt_coords[w, ids]).unsqueeze(-1)    # N x 3 x 1
            weights_mat, weights_d = convert_to_weight_matrix(match_weights[w, :, ids].T, w, T_aug)
            ones = torch.ones(weights_mat.shape).to(gpuid)

            # get R_21 and t_12_in_2
            R_21 = torch.from_numpy(solver.poses[b, w-i+1][:3, :3]).to(gpuid).unsqueeze(0)
            t_12_in_2 = torch.from_numpy(solver.poses[b, w-i+1][:3, 3:4]).to(gpuid).unsqueeze(0)
            error = points2 - (R_21 @ points1 + t_12_in_2)
            mah2_error = error.transpose(1, 2) @ weights_mat @ error

            # error threshold
            errorT = mah_thres**2
            if errorT > 0:
                ids = torch.nonzero(mah2_error.squeeze() < errorT, as_tuple=False).squeeze()
            else:
                ids = torch.arange(mah2_error.size(0))

            if ids.squeeze().nelement() <= 1:
                print('Warning: MAH threshold output has 1 or 0 elements.')
                error2 = error.transpose(1, 2) @ error
                k = min(len(error2.squeeze()), topk_backup)
                _, ids = torch.topk(error2.squeeze(), k, largest=False)

            # squared mah error
            if expect_approx_opt == 0:
                # only mean
                point_loss += torch.mean(error[ids].transpose(1, 2) @ weights_mat[ids] @ error[ids])
                unweighted_point_loss += torch.mean(error[ids].transpose(1, 2) @ ones[ids] @ error[ids])
            elif expect_approx_opt == 1:
                # sigmapoints
                Rsp = torch.from_numpy(solver.poses_sp[b, w-i, :, :3, :3]).to(gpuid).unsqueeze(1)  # s x 1 x 3 x 3
                tsp = torch.from_numpy(solver.poses_sp[b, w-i, :, :3, 3:4]).to(gpuid).unsqueeze(1)  # s x 1 x 3 x 1

                points2 = points2[ids].unsqueeze(0)  # 1 x n x 3 x 1
                points1_in_2 = Rsp @ (points1[ids].unsqueeze(0)) + tsp  # s x n x 3 x 1
                error = points2 - points1_in_2  # s x n x 3 x 1
                temp = torch.sum(error.transpose(2, 3) @ weights_mat[ids].unsqueeze(0) @ error, dim=0)/Rsp.size(0)
                unweighted_point_loss += torch.mean(error.transpose(2, 3) @ ones[ids].unsqueeze(0) @ error)
                point_loss += torch.mean(temp)
            else:
                raise NotImplementedError('Steam loss method not implemented!')

            # log det (ignore 3rd dim since it's a constant)
            logdet_loss -= torch.mean(torch.sum(weights_d[ids, 0:2], dim=1))

    # average over batches
    if bcount > 0:
        point_loss /= (bcount * (solver.window_size - 1))
        logdet_loss /= (bcount * (solver.window_size - 1))
    total_loss = point_loss + logdet_loss
    dict_loss = {'point_loss': point_loss, 'logdet_loss': logdet_loss, 'unweighted_point_loss': unweighted_point_loss}
    return total_loss, dict_loss
