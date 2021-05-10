import unittest
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from networks.steam_solver import SteamSolver
from utils.utils import get_inverse_tf, translationError, rotationError

PI = np.pi

def wrapto2pi(phi):
    if phi < 0:
        return phi + 2 * np.pi
    elif phi >= 2 * np.pi:
        return phi - 2 * np.pi
    else:
        return phi

def get_lines(vertices):
    eps = 1e-8
    N = vertices.shape[0]
    lines = np.zeros((7, N - 1))
    for i in range(N - 1):
        x1 = vertices[i, 0]
        y1 = vertices[i, 1]
        x2 = vertices[i+1, 0]
        y2 = vertices[i+1, 1]
        phi = wrapto2pi(np.arctan2(y2 - y1, x2 - x1))
        if (0 <= phi and phi < 0.25 * PI) or (0.75 * PI <= phi and phi < 1.25 * PI) or \
            (1.75 * PI <= phi and phi < 2 * PI):
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            flag = False
        else:
            m = (x2 - x1) / (y2 - y1)
            b = x1 - m * y1
            flag = True
        lines[0, i] = flag
        lines[1, i] = m
        lines[2, i] = b
        lines[3, i] = x1
        lines[4, i] = y1
        lines[5, i] = x2
        lines[6, i] = y2
    return lines

def yaw(y):
    return np.array([[np.cos(y), np.sin(y), 0], [-np.sin(y), np.cos(y), 0], [0, 0, 1]], dtype=np.float64)

class TestDistortion(unittest.TestCase):
    def test0(self):
        v = 20.0
        omega = 90.0 * np.pi / 180.0
        print(v, omega)
        square = np.array([[25, -25, -25, 25, 25], [25, 25, -25, -25, 25]]).transpose()
        plt.figure(figsize=(10, 10), tight_layout=True)
        plt.axes().set_aspect('equal')
        plt.plot(square[:, 0], square[:, 1], "k")

        lines = get_lines(square)

        x1 = []; y1 = []; x2 = []; y2 = []; a1 = []; a2 = []; t1 = []; t2 = [];

        delta_t = 0.25 / 400.0
        time = 0.0

        desc1 = np.zeros((400, 2))
        desc2 = np.zeros((400, 2))

        x_pos_vec = []; y_pos_vec = []; theta_pos_vec = [];

        for scan in range(2):
            for i in range(400):
                # Get sensor position
                theta_pos = wrapto2pi(omega * time)
                if omega == 0:
                    x_pos = v * time
                    y_pos = 0
                else:
                    x_pos = (v / omega) * np.sin(theta_pos)
                    y_pos = (v / omega) * (1 - np.cos(theta_pos))
                x_pos_vec.append(x_pos)
                y_pos_vec.append(y_pos)
                theta_pos_vec.append(theta_pos)

                theta_rad = i * 0.9 * np.pi / 180
                theta = theta_pos + theta_rad
                theta = wrapto2pi(theta)

                if scan == 0:
                    a1.append(theta_rad)
                    t1.append(time * 1e6)
                else:
                    a2.append(theta_rad)
                    t2.append(time * 1e6)

                if (0 <= theta and theta < 0.25 * PI) or (0.75 * PI <= theta and theta < 1.25 * PI) or \
                    (1.75 * PI <= theta and theta < 2 * PI):
                    m = np.tan(theta)
                    b = y_pos - m * x_pos
                    flag = False
                else:
                    m = np.cos(theta) / np.sin(theta)
                    b = x_pos - m * y_pos
                    flag = True

                dmin = 1.0e6
                x_true = 0
                y_true = 0
                eps = 1.0e-8
                for j in range(lines.shape[1]):
                    m2 = lines[1, j]
                    b2 = lines[2, j]
                    lflag = lines[0, j]
                    if flag is False and lflag == 0:
                        x_int = (b2 - b) / (m - m2)
                        y_int = m * x_int + b
                    elif flag is False and lflag == 1:
                        y_int = (m * b2 + b) / (1 - m * m2)
                        x_int = m2 * y_int + b2
                    elif flag is True and lflag == 0:
                        y_int = (m2 * b + b2) / (1 - m * m2)
                        x_int = m * y_int + b
                    else:
                        y_int = (b2 - b) / (m - m2 + eps)
                        x_int = m * y_int + b

                    if (0 <= theta and theta < PI and (y_int - y_pos) < 0) or \
                        (PI <= theta and theta < 2 * PI and (y_int - y_pos) > 0):
                        continue
                    if  (((0 <= theta and theta < 0.5 * PI) or (1.5 * PI <= theta and theta < 2 * PI)) and
                        (x_int - x_pos) < 0) or \
                        (0.5 * PI <= theta and theta < 1.5 * PI and (x_int - x_pos) > 0):
                        continue
                    x_range = [lines[3, j], lines[5, j]]
                    y_range = [lines[4, j], lines[6, j]]
                    x_range.sort()
                    y_range.sort()
                    #if x_int < x_range[0] or x_int > x_range[1] or y_int < y_range[0] or y_int > y_range[1]:
                    #    continue

                    d = (x_pos - x_int)**2 + (y_pos - y_int)**2
                    if d < dmin:
                        dmin = d
                        x_true = x_int
                        y_true = y_int

                r = np.sqrt((x_pos - x_true)**2 + (y_pos - y_true)**2)
                if scan == 0:
                    desc1[i, 0] = x_true
                    desc1[i, 1] = y_true
                    x1.append(r * np.cos(theta_rad))
                    y1.append(r * np.sin(theta_rad))
                else:
                    desc2[i, 0] = x_true
                    desc2[i, 1] = y_true
                    x2.append(r * np.cos(theta_rad))
                    y2.append(r * np.sin(theta_rad))

                time += delta_t

        plt.scatter(x1, y1, 25.0, "r")
        plt.scatter(x_pos_vec, y_pos_vec, 25.0, 'k')
        plt.scatter(x2, y2, 25.0, "b")
        plt.savefig('output.pdf', bbox_inches='tight', pad_inches=0.0)

        # Perform NN matching using the descriptors from each cloud
        kdt = KDTree(desc2, leaf_size=1, metric='euclidean')
        nnresults = kdt.query(desc1, k=1, return_distance=False)

        matches = []
        N = desc1.shape[0]
        for i in range(N):
            if nnresults[i] in matches:
                matches.append(-1)
            else:
                matches.append(nnresults[i])

        p1 = np.zeros((N, 3))
        p2 = np.zeros((N, 3))
        t1prime = np.zeros((N, 1))
        t2prime = np.zeros((N, 1))
        j = 0
        for i in range(N):
            if matches[i] == -1:
                continue
            p1[j, 0] = x1[i]
            p1[j, 1] = y1[i]
            p2[j, 0] = x2[int(matches[i])]
            p2[j, 1] = y2[int(matches[i])]
            t1prime[j, 0] = t1[i]
            t2prime[j, 0] = t2[int(matches[i])]
            j += 1
        p1.resize((j, 3))
        p2.resize((j, 3))
        p1 = np.expand_dims(p1, axis=0)
        p2 = np.expand_dims(p2, axis=0)
        p1 = torch.from_numpy(p1)
        p2 = torch.from_numpy(p2)
        # t1prime = np.expand_dims(t1prime.resize((j, 1)), axis=0)
        # t2prime = np.expand_dims(t2prime.resize((j, 1)), axis=0)

        keypoint_ints = torch.ones(1, 1, p1.shape[1])
        match_weights = torch.ones(1, 1, p1.shape[1])
        t1 = np.array(t1, dtype=np.int64).reshape((1, 400, 1))
        t2 = np.array(t2, dtype=np.int64).reshape((1, 400, 1))
        t1 = torch.from_numpy(t1)
        t2 = torch.from_numpy(t2)

        with open('config/steam.json') as f:
            config = json.load(f)
        config['window_size'] = 2
        config['gpuid'] = 'cpu'
        config['qc_diag'] = [1.0, 1.0, 1.0, 1.0, 1.0, 1]
        config['steam']['use_ransac'] = False
        config['steam']['ransac_version'] = 0
        config['steam']['use_ctsteam'] = True

        solver = SteamSolver(config)
        R_tgt_src_pred, t_tgt_src_pred = solver.optimize(p2, p1, match_weights, keypoint_ints, t2, t1)
        T_pred = np.identity(4)
        T_pred[0:3, 0:3] = R_tgt_src_pred[0, 1].cpu().numpy()
        T_pred[0:3, 3:] = t_tgt_src_pred[0, 1].cpu().numpy()
        print('T_pred:\n{}'.format(T_pred))

        T_01 = np.identity(4)
        theta_pos = omega * 0.25
        if omega == 0:
            x_pos = v * time
            y_pos = 0
        else:
            x_pos = (v / omega) * np.sin(theta_pos)
            y_pos = (v / omega) * (1 - np.cos(theta_pos))
        T_01[0:3, 0:3] = yaw(-theta_pos)
        T_01[0, 3] = x_pos
        T_01[1, 3] = y_pos
        T_10 = get_inverse_tf(T_01)
        print('T_true:\n{}'.format(T_10))

        Terr = T_01 @ T_pred
        
        t_err = translationError(Terr)
        r_err = rotationError(Terr) * 180 / np.pi

        print('t_err: {} m r_err: {} deg'.format(t_err, r_err))


if __name__ == "__main__":
    unittest.main()
