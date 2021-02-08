import unittest
import random
import numpy as np
import torch
from networks.svd import SVD
from utils.utils import convert_to_radar_frame

class TestSVD(unittest.TestCase):
    def test_basic(self):
        B = 1
        N = 100
        D = 2
        src = torch.randn(B, N, D).float()
        theta = np.pi / 4
        R_gt = torch.tensor([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]).unsqueeze(0).float()
        t_gt = torch.tensor([[10, 5]]).unsqueeze(0).float()
        out = torch.bmm(R_gt, src.transpose(2, 1)).transpose(2, 1) + t_gt
        weights = torch.ones(B, 1, N).float()

        config = {'window_size': 2, 'cart_pixel_width': 640, 'cart_resolution': 0.3456, 'gpuid': 'cpu',
                  'batch_size': 1, 'networks': {'keypoint_block': {'patch_size': 32}}}
        model = SVD(config)
        model.eval()

        R, t = model.forward(src, out, weights, convert_from_pixels=False)
        R = R.transpose(2, 1)

        R_err = torch.sum(R[:, :2, :2] - R_gt)
        t_err = torch.sum(t - t_gt)
        self.assertTrue(R_err < 1e-4)
        self.assertTrue(t_err < 1e-4)

    def test_pixel_to_cartesian(self):
        cart_pixel_width = 640
        cart_resolution = 0.3456
        patch_size = 32
        B = 1
        N = (cart_pixel_width // patch_size)**2
        src = torch.randn(B, N, 2)
        random.seed(0)
        k = 0
        for i in range(cart_pixel_width // patch_size):
            for j in range(cart_pixel_width // patch_size):
                x = random.randrange(j * patch_size, (j + 1) * patch_size)
                y = random.randrange(i * patch_size, (i + 1) * patch_size)
                src[0, k, 0] = x
                src[0, k, 1] = y
                k += 1
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
        if (cart_pixel_width % 2) != 0:
            cart_min_range = cart_pixel_width // 2 * cart_resolution
        tgt = torch.zeros(B, N, 2)
        for k in range(N):
            tgt[0, k, 1] = src[0, k, 0] * cart_resolution - cart_min_range
            tgt[0, k, 0] = cart_min_range - src[0, k, 1] * cart_resolution
        config = {'window_size': 2, 'cart_pixel_width': 640, 'cart_resolution': 0.3456, 'gpuid': 'cpu',
                  'batch_size': 1, 'networks': {'keypoint_block': {'patch_size': 32}}}
        model = SVD(config)
        model.eval()
        tgt2 = convert_to_radar_frame(src, config)
        self.assertTrue(torch.sum(tgt - tgt2) < 1e-4)

if __name__ == '__main__':
    unittest.main()
