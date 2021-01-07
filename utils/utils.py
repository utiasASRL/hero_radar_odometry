import pickle
import numpy as np
import torch

def supervised_loss(R_tgt_src_pred, t_tgt_src_pred, batch, config):
    T_21 = batch['T_21'].to(config['gpuid'])
    # Get ground truth transforms
    T_tgt_src = T_21[::config['window_size']]
    R_tgt_src = T_tgt_src[:, :3, :3]
    t_tgt_src = T_tgt_src[:, :3, 3].unsqueeze(-1)
    svd_loss, R_loss, t_loss = SVD_loss(R_tgt_src, R_tgt_src_pred, t_tgt_src, t_tgt_src_pred, config['gpuid'])
    dict_loss = {'R_loss': R_loss, 't_loss': t_loss}
    return svd_loss, dict_loss

def SVD_loss(R, R_pred, t, t_pred, gpuid='cpu', alpha=10.0):
    batch_size = R.size(0)
    identity = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(gpuid)
    loss_fn = torch.nn.L1Loss()
    R_loss = alpha * loss_fn(torch.matmul(R_pred.transpose(2, 1), R), identity)
    # R_loss = alpha * loss_fn(R_pred @ R, identity)
    t_loss = 1.0 * loss_fn(t_pred, t)
    svd_loss = R_loss + t_loss
    return svd_loss, R_loss, t_loss

def get_inverse_tf(T):
    """Returns the inverse of a given 4x4 homogeneous transform."""
    T2 = np.identity(4, dtype=np.float32)
    R = T[0:3, 0:3]
    t = T[0:3, 3].reshape(3, 1)
    T2[0:3, 0:3] = R.transpose()
    t = np.matmul(-1 * R.transpose(), t)
    T2[0:3, 3] = np.squeeze(t)
    return T2

def get_transform(x, y, theta):
    """Returns a 4x4 homogeneous 3D transform for a given 2D (x, y, theta)."""
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    T = np.identity(4, dtype=np.float32)
    T[0:2, 0:2] = R
    T[0, 3] = x
    T[1, 3] = y
    return T

def get_transform2(R, t):
    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t.squeeze()
    return T

def enforce_orthog(T):
    """Enforces the orthogonality of a 3x3 rotation matrix within a 4x4 homogeneous transformation matrix."""
    if abs(np.linalg.det(T[0:3, 0:3]) - 1) < 1e-10:
        return T
    c1 = T[0:3, 1]
    c2 = T[0:3, 2]
    c1 /= np.linalg.norm(c1)
    c2 /= np.linalg.norm(c2)
    newcol0 = np.cross(c1, c2)
    newcol1 = np.cross(c2, newcol0)
    T[0:3, 0] = newcol0
    T[0:3, 1] = newcol1
    T[0:3, 2] = c2
    return T

# Use axis-angle representation to get a single number for rotation error
def rotationError(T):
    return abs(np.arcsin(T[0, 1]))  # SO(2)

def translationError(T):
    return np.sqrt(T[0, 3]**2 + T[1, 3]**2 + T[2, 3]**2)

def computeMedianError(T_gt, R_pred, t_pred):
    """Computes the median translation and rotation error along with their standard deviations."""
    t_error = []
    r_error = []
    for i, T in enumerate(T_gt):
        T_pred = get_transform2(R_pred[i], t_pred[i])
        T_error = np.matmul(T, get_inverse_tf(T_pred))
        t_error.append(translationError(T_error))
        r_error.append(180 * rotationError(T_error) / np.pi)
    t_error = np.array(t_error)
    r_error = np.array(r_error)
    return [np.median(t_error), np.std(t_error), np.median(r_error), np.std(r_error)]

def trajectoryDistances(poses):
    """Calculates path length along the trajectory."""
    dist = [0]
    for i in range(1, len(poses)):
        P1 = get_inverse_tf(poses[i - 1])
        P2 = get_inverse_tf(poses[i])
        dx = P1[0, 3] - P2[0, 3]
        dy = P1[1, 3] - P2[1, 3]
        dist.append(dist[i-1] + np.sqrt(dx**2 + dy**2))
    return dist

def lastFrameFromSegmentLength(dist, first_frame, length):
    for i in range(first_frame, len(dist)):
        if dist[i] > dist[first_frame] + length:
            return i
    return -1

def calcSequenceErrors(poses_gt, poses_pred):
    lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    err = []
    step_size = 4  # Every second
    # Pre-compute distances from ground truth as reference
    dist = trajectoryDistances(poses_gt)

    for first_frame in range(0, len(poses_gt), step_size):
        for length in lengths:
            last_frame = lastFrameFromSegmentLength(dist, first_frame, length)
            if last_frame == -1:
                continue
            # Compute rotational and translation errors
            pose_delta_gt = np.matmul(poses_gt[last_frame], get_inverse_tf(poses_gt[first_frame]))
            pose_delta_res = np.matmul(poses_pred[last_frame], get_inverse_tf(poses_pred[first_frame]))
            pose_error = np.matmul(pose_delta_gt, get_inverse_tf(pose_delta_res))
            r_err = rotationError(pose_error)
            t_err = translationError(pose_error)
            # Approx speed
            num_frames = float(last_frame - first_frame + 1)
            speed = float(length) / (0.25 * num_frames)
            err.append([first_frame, r_err/float(length), t_err/float(length), length, speed])
    return err

def getStats(err):
    t_err = 0
    r_err = 0
    for e in err:
        t_err += e[2]
        r_err += e[1]
    t_err /= float(len(err))
    r_err /= float(len(err))
    return t_err, r_err

def computeKittiMetrics(T_gt, R_pred, t_pred, seq_len):
    """Computes the translational (%) and rotational drift (deg/m) in the KITTI style."""
    seq_indices = []
    idx = 0
    for s in seq_len:
        seq_indices.append(list(range(idx, idx + s - 1)))
        idx += (s - 1)
    err = []
    for indices in seq_indices:
        T_gt_ = np.identity(4)
        T_pred_ = np.identity(4)
        poses_gt = []
        poses_pred = []
        for i in indices:
            T_gt_ = np.matmul(T_gt[i], T_gt_)
            T_pred_ = np.matmul(get_transform2(R_pred[i], t_pred[i]), T_pred_)
            enforce_orthog(T_gt_)
            enforce_orthog(T_pred_)
            poses_gt.append(T_gt_)
            poses_pred.append(T_pred_)
        err.extend(calcSequenceErrors(poses_gt, poses_pred))
    t_err, r_err = getStats(err)
    return t_err * 100, r_err * 180 / np.pi, err

def saveKittiErrors(err, fname):
    pickle.dump(err, open(fname, 'wb'))

def loadKittiErrors(fname):
    return pickle.load(open(fname, 'rb'))

def save_in_yeti_format(T_gt, R_pred, t_pred, timestamps, seq_len, seq_names, root='./'):
    """This function converts outputs to a format that is backwards compatible with the yeti repository."""

    seq_indices = []
    idx = 0
    for s in seq_len:
        seq_indices.append(list(range(idx, idx + s - 1)))
        idx += (s - 1)

    for s, indices in enumerate(seq_indices):
        fname = root + 'accuracy' + seq_names[s] + '.csv'
        with open(fname, 'w') as f:
            f.write('x,y,yaw,gtx,gty,gtyaw,time1,time2\n')
            for i in indices:
                yaw = -1 * np.arcsin(R_pred[i][0, 1])
                gtyaw = -1 * np.arcsin(T_gt[i][0, 1])
                t = np.matmul(-1 * R_pred[i].transpose(), np.reshape(t_pred[i], (3, 1)))
                T = get_inverse_tf(T_gt[i])
                f.write('{},{},{},{},{},{},{},{}\n'.format(t[0, 0], t[1, 0], yaw, T[0, 3], T[1, 3], gtyaw,
                                                           timestamps[i][0], timestamps[i][1]))

def convert_to_radar_frame(pixel_coords, cart_pixel_width, cart_resolution, gpuid):
    """Converts pixel_coords (B x N x 2) from pixel coordinates to metric coordinates in the radar frame."""
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    B, N, _ = pixel_coords.size()
    R = torch.tensor([[0, -cart_resolution], [cart_resolution, 0]]).expand(B, 2, 2).to(gpuid)
    t = torch.tensor([[cart_min_range], [-cart_min_range]]).expand(B, 2, N).to(gpuid)
    return (torch.bmm(R, pixel_coords.transpose(2, 1)) + t).transpose(2, 1)
