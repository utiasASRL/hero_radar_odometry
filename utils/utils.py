import pickle
import numpy as np
import torch
import torch.nn.functional as F

def get_inverse_tf(T):
    """Returns the inverse of a given 4x4 homogeneous transform."""
    T2 = np.identity(4, dtype=np.float32)
    R = T[0:3, 0:3]
    t = T[0:3, 3].reshape(3, 1)
    T2[0:3, 0:3] = R.transpose()
    T2[0:3, 3:] = np.matmul(-1 * R.transpose(), t)
    return T2

def get_transform(x, y, theta):
    """Returns a 4x4 homogeneous 3D transform for a given 2D (x, y, theta)."""
    T = np.identity(4, dtype=np.float32)
    T[0:2, 0:2] = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    T[0, 3] = x
    T[1, 3] = y
    return T

def get_transform2(R, t):
    T = np.identity(4, dtype=np.float32)
    T[0:3, 0:3] = R
    T[0:3, 3] = t.squeeze()
    return T

def enforce_orthog(T, dim=3):
    """Enforces the orthogonality of a 3x3 rotation matrix within a 4x4 homogeneous transformation matrix."""
    if dim == 2:
        if abs(np.linalg.det(T[0:2, 0:2]) - 1) < 1e-10:
            return T
        R = T[0:2, 0:2]
        epsilon = 0.001
        if abs(R[0, 0] - R[1, 1]) > epsilon or abs(R[1, 0] + R[0, 1]) > epsilon:
            print("WARNING: this is not a proper rigid transformation:", R)
            return T
        a = (R[0, 0] + R[1, 1]) / 2
        b = (-R[1, 0] + R[0, 1]) / 2
        sum = np.sqrt(a**2 + b**2)
        a /= sum
        b /= sum
        R[0, 0] = a; R[0, 1] = b
        R[1, 0] = -b; R[1, 1] = a
        T[0:2, 0:2] = R
    if dim == 3:
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

def carrot(xbar):
    x = xbar.squeeze()
    if x.shape[0] == 3:
        return np.array([[0, -x[2], x[1]],
                         [x[2], 0, -x[0]],
                         [-x[1], x[0], 0]])
    elif x.shape[0] == 6:
        return np.array([[0, -x[5], x[4], x[0]],
                         [x[5], 0, -x[3], x[1]],
                         [-x[4], x[3], 0, x[2]],
                         [0, 0, 0, 1]])
    else:
        print('WARNING: attempted carrot operator on invalid vector shape')
        return xbar

def se3ToSE3(xi):
    """Lie Vector xi = [rho, phi]^T (6 x 1) --> SE(3) T = [C, R; 0 0 0 1] (4 x 4)"""
    T = np.identity(4, dtype=np.float32)
    rho = xi[0:3].reshape(3, 1)
    phibar = xi[3:6].reshape(3, 1)
    phi = np.linalg.norm(phibar)
    R = np.identity(3)
    if phi != 0:
        phibar /= phi  # normalize
        I = np.identity(3)
        R = np.cos(phi) * I + (1 - np.cos(phi)) * phibar @ phibar.T + np.sin(phi) * carrot(phibar)
        J = I * np.sin(phi) / phi + (1 - np.sin(phi) / phi) * phibar @ phibar.T + \
            carrot(phibar) * (1 - np.cos(phi)) / phi
        rho = J @ rho
    T[0:3, 0:3] = R
    T[0:3, 3:] = rho
    return T

def SE3tose3(T):
    """SE(3) T = [C, R; 0 0 0 1] (4 x 4) --> Lie Vector xi = [rho, phi]^T (6 x 1)"""
    R = T[0:3, 0:3]
    evals, evecs = np.linalg.eig(R)
    idx = -1
    for i in range(3):
        if evals[i].real != 0 and evals[i].imag == 0:
            idx = i
            break
    assert(idx != -1)
    abar = evecs[idx].real.reshape(3, 1)
    phi = np.arccos((np.trace(R) - 1) / 2)
    rho = T[0:3, 3:]
    if phi != 0:
        I = np.identity(3)
        J = I * np.sin(phi) / phi + (1 - np.sin(phi) / phi) * abar @ abar.T + \
                carrot(abar) * (1 - np.cos(phi)) / phi
        rho = np.linalg.inv(J) @ rho
    xi = np.zeros((6, 1))
    xi[0:3, 0:] = rho
    xi[3:, 0:] = phi * abar
    return xi

# Use axis-angle representation to get a single number for rotation error
def rotationError(T):
    d = 0.5 * (np.trace(T[0:3, 0:3]) - 1)
    return np.arccos(max(min(d, 1.0), -1.0))

def translationError(T, dim=2):
    if dim == 2:
        return np.sqrt(T[0, 3]**2 + T[1, 3]**2)
    if dim == 3:
        return np.sqrt(T[0, 3]**2 + T[1, 3]**2 + T[2, 3]**2)

def computeMedianError(T_gt, T_pred):
    """Computes the median translation and rotation error along with their standard deviations."""
    t_error = []
    r_error = []
    for i, T in enumerate(T_gt):
        T_error = np.matmul(T, get_inverse_tf(T_pred[i]))
        t_error.append(translationError(T_error))
        r_error.append(180 * rotationError(T_error) / np.pi)
    t_error = np.array(t_error)
    r_error = np.array(r_error)
    return [np.median(t_error), np.std(t_error), np.median(r_error), np.std(r_error), np.mean(t_error),
            np.mean(r_error), t_error, r_error]

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

def computeKittiMetrics(T_gt, T_pred, seq_lens):
    """
        Computes the translational (%) and rotational drift (deg/m) in the KITTI style.
        T_gt: List of 4x4 homogeneous transforms (Frame t to Frame t+1)
        T_pred: List of 4x4 homogeneous transforms (Frame t to Frame t+1)
        seq_lens: List of sequence lengths
    """
    seq_indices = []
    idx = 0
    for s in seq_lens:
        seq_indices.append(list(range(idx, idx + s - 1)))
        idx += (s - 1)
    err_list = []
    for indices in seq_indices:
        T_gt_ = np.identity(4)
        T_pred_ = np.identity(4)
        poses_gt = []
        poses_pred = []
        for i in indices:
            T_gt_ = np.matmul(T_gt[i], T_gt_)
            T_pred_ = np.matmul(T_pred[i], T_pred_)
            enforce_orthog(T_gt_)
            enforce_orthog(T_pred_)
            poses_gt.append(T_gt_)
            poses_pred.append(T_pred_)
        err = calcSequenceErrors(poses_gt, poses_pred)
        t_err, r_err = getStats(err)
        err_list.append([t_err, r_err])
    err_list = np.asarray(err_list)
    avg = np.mean(err_list, axis=0)
    t_err = avg[0]
    r_err = avg[1]
    return t_err * 100, r_err * 180 / np.pi, err

def saveKittiErrors(err, fname):
    pickle.dump(err, open(fname, 'wb'))

def loadKittiErrors(fname):
    return pickle.load(open(fname, 'rb'))

def save_in_yeti_format(T_gt, T_pred, timestamps, seq_lens, seq_names, root='./'):
    """This function converts outputs to a format that is backwards compatible with the yeti repository."""
    seq_indices = []
    idx = 0
    for s in seq_lens:
        seq_indices.append(list(range(idx, idx + s - 1)))
        idx += (s - 1)

    for s, indices in enumerate(seq_indices):
        fname = root + 'accuracy' + seq_names[s] + '.csv'
        with open(fname, 'w') as f:
            f.write('x,y,yaw,gtx,gty,gtyaw,time1,time2\n')
            for i in indices:
                R_pred = T_pred[i][:3, :3]
                t_pred = T_pred[i][:3, 3:]
                yaw = -1 * np.arcsin(R_pred[0, 1])
                gtyaw = -1 * np.arcsin(T_gt[i][0, 1])
                t = np.matmul(-1 * R_pred.transpose(), np.reshape(t_pred, (3, 1)))
                T = get_inverse_tf(T_gt[i])
                f.write('{},{},{},{},{},{},{},{}\n'.format(t[0, 0], t[1, 0], yaw, T[0, 3], T[1, 3], gtyaw,
                                                           timestamps[i][0], timestamps[i][1]))

def load_icra21_results(results_loc, seq_names, seq_lens):
    T_icra = []
    for i, seq_name in enumerate(seq_names):
        fname = results_loc + 'accuracy' + seq_name + '.csv'
        with open(fname, 'r') as f:
            f.readline()  # Clear out the header
            lines = f.readlines()
            count = 0
            for line in lines:
                line = line.split(',')
                # Retrieve the transform estimated by MC-RANSAC + DOPPLER compensation
                T_icra.append(get_inverse_tf(get_transform(float(line[11]), float(line[12]), float(line[13]))))
                count += 1
            # Append identity transforms at the end in case the ICRA results ended early by a couple frames
            if count < seq_lens[i]:
                print('WARNING: ICRA results shorter than seq_len by {}. Append last TF.'.format((seq_lens[i] - count)))
            while count < seq_lens[i]:
                T_icra.append(T_icra[-1])
                count += 1
    return T_icra

def normalize_coords(coords_2D, width, height):
    """Normalizes coords_2D (BW x N x 2) to be within [-1, 1] """
    batch_size = coords_2D.size(0)
    u_norm = (2 * coords_2D[:, :, 0].reshape(batch_size, -1) / (width - 1)) - 1
    v_norm = (2 * coords_2D[:, :, 1].reshape(batch_size, -1) / (height - 1)) - 1
    return torch.stack([u_norm, v_norm], dim=2)  # BW x num_patches x 2

def convert_to_radar_frame(pixel_coords, config):
    cart_pixel_width = config['cart_pixel_width']
    cart_resolution = config['cart_resolution']
    gpuid = config['gpuid']
    """Converts pixel_coords (B x N x 2) from pixel coordinates to metric coordinates in the radar frame."""
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    B, N, _ = pixel_coords.size()
    R = torch.tensor([[0, -cart_resolution], [cart_resolution, 0]]).expand(B, 2, 2).to(gpuid)
    t = torch.tensor([[cart_min_range], [-cart_min_range]]).expand(B, 2, N).to(gpuid)
    return (torch.bmm(R, pixel_coords.transpose(2, 1)) + t).transpose(2, 1)

# Used with UnderTheRadar
def get_indices(batch_size, window_size):
    src_ids = []
    tgt_ids = []
    for i in range(batch_size):
        for j in range(window_size - 1):
            idx = i * window_size + j
            src_ids.append(idx)
            tgt_ids.append(idx + 1)
    return src_ids, tgt_ids

# Used with HERO
def get_indices2(batch_size, window_size, asTensor=False):
    src_ids = []
    tgt_ids = []
    for i in range(batch_size):
        idx = i * window_size
        for j in range(idx + 1, idx + window_size):
            tgt_ids.append(j)
            src_ids.append(idx)
    if asTensor:
        src_ids = np.asarray(src_ids, dtype=np.int64)
        tgt_ids = np.asarray(tgt_ids, dtype=np.int64)
        return torch.from_numpy(src_ids), torch.from_numpy(tgt_ids)
    else:
        return src_ids, tgt_ids

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_T_ba(out, a, b):
    T_b0 = np.eye(4)
    T_b0[:3, :3] = out['R'][0, b].detach().cpu().numpy()
    T_b0[:3, 3:4] = out['t'][0, b].detach().cpu().numpy()
    T_a0 = np.eye(4)
    T_a0[:3, :3] = out['R'][0, a].detach().cpu().numpy()
    T_a0[:3, 3:4] = out['t'][0, a].detach().cpu().numpy()
    return np.matmul(T_b0, get_inverse_tf(T_a0))

def convert_to_weight_matrix(w, window_id, T_aug=[]):
    """
        w: n_points x S
        This function converts the S-dimensional weights estimated for each keypoint into
        a 2x2 weight (inverse covariance) matrix for each keypoint.
        If S = 1, Wout = diag(exp(w), exp(w), 1e4)
        If S = 3, use LDL^T to obtain 2x2 covariance, place on top-LH corner. 1e4 bottom-RH corner.
    """
    z_weight = 9.2103  # 9.2103 = log(1e4), 1e4 is inverse variance of 1cm std dev
    if w.size(1) == 1:
        # scalar weight
        A = torch.zeros(w.size(0), 9, device=w.device)
        A[:, (0, 4)] = torch.exp(w)
        A[:, 8] = torch.exp(torch.tensor(z_weight))
        A = A.reshape((-1, 3, 3))
        d = torch.zeros(w.size(0), 3, device=w.device)
        d[:, 0:2] += w
        d[:, 2] += z_weight
    elif w.size(1) == 3:
        # 2x2 matrix
        L = torch.zeros(w.size(0), 4, device=w.device)
        L[:, (0, 3)] = 1
        L[:, 2] = w[:, 0]
        L = L.reshape((-1, 2, 2))
        D = torch.zeros(w.size(0), 4, device=w.device)
        D[:, (0, 3)] = torch.exp(w[:, 1:])
        D = D.reshape((-1, 2, 2))
        A2x2 = L @ D @ L.transpose(1, 2)

        if T_aug:  # if list is not empty
            Rot = T_aug[window_id].to(w.device)[:2, :2].unsqueeze(0)
            A2x2 = Rot.transpose(1, 2) @ A2x2 @ Rot

        A = torch.zeros(w.size(0), 3, 3, device=w.device)
        A[:, 0:2, 0:2] = A2x2
        A[:, 2, 2] = torch.exp(torch.tensor(z_weight))
        d = torch.ones(w.size(0), 3, device=w.device)*z_weight
        d[:, 0:2] = w[:, 1:]
    else:
        assert False, "Weight scores should be dim 1 or 3"

    return A, d

def mask_intensity_filter(data, patch_size, patch_mean_thres=0.05):
    """ Given a cartesian mask of likely target pixels (data), this function computes the percentage of
        likely target pixels in a given square match of the input. The output is a list of booleans indicate whether
        each patch either has more (True) or less (False) % likely target pixels than the patch_mean_thres.
    """
    int_patches = F.unfold(data, kernel_size=patch_size, stride=patch_size)
    keypoint_int = torch.mean(int_patches, dim=1, keepdim=True)  # BW x 1 x num_patches
    return keypoint_int >= patch_mean_thres

def wrapto2pi(phi):
    """Ensures that the output angle phi is within the interval [0, 2*pi)"""
    if phi < 0:
        return phi + 2 * np.pi * np.ceil(phi / (-2 * np.pi))
    elif phi >= 2 * np.pi:
        return (phi / (2 * np.pi) % 1) * 2 * np.pi
    else:
        return phi

def getApproxTimeStamps(points, times):
    """ Retrieves the approximate timestamp of each target point
        points: List of np.array() (N, 2)
        times: List of np.array() (400,) * time at each azimuth
        Out: List of np.array() (N,)
    """
    azimuth_step = (2 * np.pi) / 400
    timestamps = []
    BW = len(points)  # batch * window_size
    for i in range(len(points)):
        p = points[i]
        ptimes = times[i]
        point_times = []
        for k in range(p.shape[0]):
            x = p[k, 0]
            y = p[k, 1]
            phi = np.arctan2(y, x)
            phi = wrapto2pi(phi)
            time_idx = phi / azimuth_step
            t1 = ptimes[int(np.floor(time_idx))]
            idx2 = int(np.ceil(time_idx))
            t2 = ptimes[idx2 if idx2 < 400 else 399]
            # interpolate to get slightly more precise timestamp
            ratio = time_idx % 1
            t = int(t1 + ratio * (t2 - t1))
            point_times.append(t)
        timestamps.append(np.array(point_times))
    return timestamps

def undistort_pointcloud(points, point_times, t_refs, solver):
    """ Removes motion distortion from pointclouds
        points: List of np.array() (N, 2-4)
        point_times: List of np.array() (N,) * time at each point
        t_refs: A reference time for each pointcloud, transform each point into the sensor frame at this time.
    """
    for i in range(len(points)):
        p = points[i]
        ptimes = point_times[i]
        t_ref = t_refs[i]
        for j in range(M):
            T_0a = np.identity(4, dtype=np.float32)
            solver.getPoseBetweenTimes(T_0a, ptimes[j], t_ref)
            pbar = T_0a @ p[j].reshape(4, 1)
            p[j, :] = pbar[:]
        points[i] = p
    return points
