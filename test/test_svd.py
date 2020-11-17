import numpy as np
import torch
import networks
from networks.svd import SVD
import unittest

class TestSVD(unittest.TestCase):
    def test_basic(self):
        B = 1
        N = 100
        D = 2
        src = torch.randn(B, N, D)
        theta = np.pi / 4
        R_gt = torch.tensor([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]]).unsqueeze(0)
        t_gt = torch.tensor([[10, 5]]).unsqueeze(0)
        out = torch.bmm(R_gt, src.transpose(2,1)).transpose(2, 1) + t_gt
        weights = torch.ones(B, 1, N)

        config = {'window_size': 2}
        model = SVD(config)
        model.eval()

        R, t = model.forward(src, out, weights)

        R_err = torch.sum(R[:,:2,:2] - R_gt)
        t_err = torch.sum(t - t_gt)
        self.assertTrue(R_err < 1e-4)
        self.assertTrue(t_err < 1e-4)

if __name__ == '__main__':
    unittest.main()
