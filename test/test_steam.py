import unittest
import random
import numpy as np
import torch
import cpp.build.steampy as steampy
import cpp.build.SteamSolver as SteamCpp

class TestSteam(unittest.TestCase):
    def test_basic(self):
        N = 100
        D = 3
        src = torch.randn(D, N)
        theta = np.pi / 8
        R_gt = torch.eye(3)
        R_gt[:2, :2] = torch.tensor([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
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

        R_err = torch.sum(R - R_gt)
        t_err = torch.sum(t - t_gt)
        self.assertTrue(R_err < 1e-4)
        self.assertTrue(t_err < 1e-4)

    def test_solver_class(self):
        N = 100
        D = 2
        src = torch.randn(D, N)
        theta = np.pi / 8
        R_gt = torch.eye(2)
        R_gt[:2, :2] = torch.tensor([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
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

        R_err = torch.sum(R - R_gt)
        t_err = torch.sum(t - t_gt)
        self.assertTrue(R_err < 1e-4)
        self.assertTrue(t_err < 1e-4)


if __name__ == '__main__':
    unittest.main()
