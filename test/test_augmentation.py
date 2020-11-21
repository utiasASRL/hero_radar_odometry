import unittest
import numpy as np
import torch
from datasets.transforms import augmentBatch
from utils.utils import get_inverse_tf

def convert_to_metric(pixel_coords, cart_resolution, cart_pixel_width):
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    metric_coords = []
    for u, v in pixel_coords:
        x = u * cart_resolution - cart_min_range
        y = cart_min_range - v * cart_resolution
        metric_coords.append([x, y])
    return metric_coords

class TestAugmentation(unittest.TestCase):
    def test_basic(self):
        # Create a test image
        cart_width = 100
        cart_res = 0.25
        img = np.zeros((cart_width, cart_width), dtype=np.float32)
        src_coords = [[24, 74], [24, 24], [74, 24], [74, 74]]
        for u, v in src_coords:
            img[u, v] = 1
        torch_img = torch.from_numpy(img)
        torch_img = torch_img.expand(2, 1, cart_width, cart_width)

        config = {'augmentation': {'rot_max': np.pi / 4}}
        T = np.identity(4, dtype=np.float32).reshape(1, 4, 4)
        T2 = np.identity(4, dtype=np.float32).reshape(1, 4, 4)
        T3 = np.concatenate((T, T2), axis=0)
        T = torch.from_numpy(T3)
        batch = {'data': torch_img, 'T_21': T}

        np.random.seed(1234)
        batch = augmentBatch(batch, config)
        out = batch['data'][1].numpy().squeeze()
        T_out = batch['T_21'][0].numpy()
        coords = out.nonzero()
        out_coords = []
        for u, v in zip(coords[0], coords[1]):
            out_coords.append([u, v])

        src_metric = convert_to_metric(src_coords, cart_res, cart_width)
        out_metric = convert_to_metric(out_coords, cart_res, cart_width)

        outliers = 0
        for x, y in out_metric:
            outlier = 1
            xbar = np.array([x, y, 0, 1]).reshape(4, 1)
            xbar = np.matmul(get_inverse_tf(T_out), xbar)
            xn = xbar[0]
            yn = xbar[1]
            for xs, ys in src_metric:
                if np.sqrt((xn - xs)**2 + (yn - ys)**2) <= cart_res * 3:
                    outlier = 0
                    break
            outliers += outlier
        self.assertTrue(outliers == 0)

if __name__ == '__main__':
    unittest.main()
