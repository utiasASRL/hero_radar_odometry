import unittest
import random
import torch
import numpy as np
from networks.softmax_matcher import SoftmaxMatcher

class TestMatching(unittest.TestCase):
    def test_basic(self):
        B = 2
        N = 1024
        C = 248
        H = 512
        W = 512

        src_scores = torch.ones(B, 1, N)
        src_desc = torch.randn(B, C, N)
        tgt_scores_dense = torch.ones(B, 1, H, W)
        tgt_desc_dense = torch.randn(B, C, H, W)

        # "Hide" the src descriptors within the dense target descriptor map
        L = list(range(0, H * W))
        random.seed(0)
        random.shuffle(L)
        L = L[:N]
        row_col = [(index % H, index // H) for index in L]
        for i in range(N):
            x = src_desc[0, :, i]
            x = x.reshape(1, C)
            tgt_desc_dense[1, :, row_col[i][0], row_col[i][1]] = x
        # Now we have known correspondences: i <--> row_col

        config = {'networks': {'matcher_block': {'softmax_temp': 0.01}}}
        config['window_size'] = 2
        config['gpuid'] = "cpu"
        model = SoftmaxMatcher(config)
        model.eval()

        pseudo_coords, _ = model.forward(src_scores, src_desc, tgt_scores_dense, tgt_desc_dense)

        # Now we can check whether the pseudo coordinates are the same as our known correspondences:
        outliers = 0
        outlier_threshold = 1e-4
        for i in range(N):
            dx = pseudo_coords[0, i, 0].item() - row_col[i][1]
            dy = pseudo_coords[0, i, 1].item() - row_col[i][0]
            d = np.sqrt(dx**2 + dy**2)
            if d > outlier_threshold:
                outliers += 1
        print('outliers: {}'.format(outliers))

        self.assertTrue(outliers / N < 0.001)

if __name__ == '__main__':
    unittest.main()
