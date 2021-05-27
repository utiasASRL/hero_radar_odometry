import unittest
import json
import numpy as np
import torch
import cpp.build.steampy as steampy
import cpp.build.SteamSolver as SteamCpp
from networks.steam_solver import SteamSolver
from utils.utils import get_inverse_tf, rotationError, translationError

def get_times(t0, M=400):
    times = torch.zeros(M, dtype=torch.int64)
    delta_t = int(1e6 * 0.25 / M)
    for i in range(M):
        times[i] = int(t0 + i * delta_t)
    return times

def get_T_ba(R_pred, t_pred, a, b):
    T_b0 = np.eye(4)
    T_b0[:3, :3] = R_pred[0, b].numpy()
    T_b0[:3, 3:4] = t_pred[0, b].numpy()
    T_a0 = np.eye(4)
    T_a0[:3, :3] = R_pred[0, a].numpy()
    T_a0[:3, 3:4] = t_pred[0, a].numpy()
    return np.matmul(T_b0, get_inverse_tf(T_a0))

class TestSteam(unittest.TestCase):
    def test_basic(self):
        N = 100
        src = torch.randn(3, N)
        theta = np.pi / 8
        R_gt = torch.eye(3)
        R_gt[:2, :2] = torch.tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        t_gt = torch.tensor([[1], [2], [1]])
        out = R_gt @ src + t_gt

        # points must be list of N x 3
        p2_list = [out.T.detach().cpu().numpy()]
        p1_list = [src.T.detach().cpu().numpy()]

        # weights must be list of N x 3 x 3
        w_list = [torch.eye(3).repeat(N,1,1).detach().cpu().numpy()]

        # poses are window_size x (num sigmapoints + 1) x 4 x 4
        # vels are window_size x 6
        # num sigmapoints is 12
        window_size = 2
        poses = torch.eye(4).unsqueeze(0).repeat(window_size,1,1,1).detach().cpu().numpy()
        vels = torch.zeros(window_size, 6).detach().cpu().numpy()

        # run steam
        dt = 1.0    # timestep for motion prior
        sigmapoints = False
        steampy.run_steam(p2_list, p1_list, w_list, poses, vels, sigmapoints, dt)

        # 2nd pose will be T_21
        R = torch.from_numpy(poses[1, 0, :3, :3])
        t = torch.from_numpy(poses[1, 0, :3, 3:])

        r_err = rotationError(R, R_gt)
        t_err = translationError(t, t_gt)
        self.assertTrue(r_err < 1e-4, "Rotation: {} != {}".format(R, R_gt))
        self.assertTrue(t_err < 1e-4, "Translation: {} != {}".format(t, t_gt))

    def test_solver_class(self):
        N = 100
        src = torch.randn(2, N)
        theta = np.pi / 8
        R_gt = torch.eye(2)
        R_gt[:2, :2] = torch.tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        t_gt = torch.tensor([[1], [2]])
        out = R_gt @ src + t_gt
        zeros_vec = np.zeros((N, 1), dtype=np.float32)

        # points must be list of N x 3
        points2 = out.T.detach().cpu().numpy()
        points1 = src.T.detach().cpu().numpy()

        # weights must be list of N x 3 x 3
        identity_weights = np.tile(np.expand_dims(np.eye(3, dtype=np.float32), 0), (N, 1, 1))

        # poses are window_size x 4 x 4
        window_size = 2
        poses = np.tile(
            np.expand_dims(np.eye(4, dtype=np.float32), 0),
            (window_size, 1, 1))

        # run steam
        dt = 0.25
        solver = SteamCpp.SteamSolver(dt, window_size, False)
        solver.setMeas([np.concatenate((points2, zeros_vec), 1)],
                       [np.concatenate((points1, zeros_vec), 1)], [identity_weights])
        solver.optimize()

        # get pose output
        solver.getPoses(poses)

        # 2nd pose will be T_21
        R = torch.from_numpy(poses[1, :2, :2])
        t = torch.from_numpy(poses[1, :2, 3:])

        r_err = rotationError(R, R_gt)
        t_err = translationError(t, t_gt)
        self.assertTrue(r_err < 1e-4, "Rotation: {} != {}".format(R, R_gt))
        self.assertTrue(t_err < 1e-4, "Translation: {} != {}".format(t, t_gt))

    def test_Tsv(self):
        with open('config/steam.json', 'r') as f:
            config = json.load(f)
        # initialize solver
        solver = SteamSolver(config)

        # create test data
        N = 100
        src = torch.randn(2, N)
        theta = np.pi / 8
        R_gt = torch.tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        t_gt = torch.tensor([[1], [2]])
        out = R_gt @ src + t_gt

        zeropad = torch.nn.ZeroPad2d((0, 1, 0, 0))
        points1 = zeropad(src.T).unsqueeze(0)  # pseudo
        points2 = zeropad(out.T).unsqueeze(0)  # keypoint

        if self.config['flip_y']:
            points1[:, :, 1] *= -1.0
            points2[:, :, 1] *= -1.0

        t0 = 0
        t1 = 250000
        t2 = 500000
        times_src = get_times(t0).reshape(1, 1, 400)
        times_tgt = get_times(t1).reshape(1, 1, 400)
        t_ref_src = torch.tensor([0, t1]).reshape(1, 1, 2)
        t_ref_tgt = torch.tensor([t1, t2]).reshape(1, 1, 2)

        keypoint_ints = torch.ones((1, 1, N))
        match_weights = torch.ones((1, 1, N))

        R_out, t_out = solver.optimize(points2, points1, match_weights, keypoint_ints,
                                         time_tgt, time_src, t_ref_tgt, t_ref_src)

        T_pred = get_T_ba(R_out, t_out, 0, 1)
        R = T_pred[:3, :3]
        t = T_pred[:3, 3:]

        r_err = rotationError(R, R_gt)
        t_err = translationError(t, t_gt)
        self.assertTrue(r_err < 1e-4, "Rotation: {} != {}".format(R, R_gt))
        self.assertTrue(t_err < 1e-4, "Translation: {} != {}".format(t, t_gt))

if __name__ == '__main__':
    unittest.main()
