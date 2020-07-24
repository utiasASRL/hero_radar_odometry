# sys imports
import os
import sys

# third-party imports
import pykitti
import numpy as np
from PIL import Image

# project imports
from utils.config import KittiConfig
from utils.helper_func import pc2img, pc2img_slow

# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':
    test_category = {
        'PC2IMG_MIG': False,
        'PC2IMG': True,
    }

    ##########################
    # UNIT TEST ON PC2IMG FUNC
    ##########################
    if test_category['PC2IMG_MIG']:
        config = KittiConfig()

        base_dir = config.base_dir
        seq_num = '00'
        dataset = pykitti.odometry(base_dir, seq_num)

        # take a sample point cloud
        num_poses = len(dataset.poses)
        np.random.seed(0)
        sample_n = 1
        sample_idx = np.random.randint(0, num_poses, sample_n)
        sample_velo = dataset.get_velo(sample_idx[0])

        # convert to image
        velo_img, velo_range = pc2img(sample_velo, config, debug=True)

        # write to disk for verification
        save_dir = '/home/haowei/MEGA/Research/src/GridConv/results/7-24/v0'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        im = Image.fromarray(velo_range)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save("{}/velo.jpeg".format(save_dir))
        print('==============================')
        print('======IMG SAVED TO DISK=======')
        print('==============================')

    if test_category['PC2IMG']:
        print('==============================')
        print('======UNIT TEST PC2IMG========')
        print('==============================')
        config = KittiConfig()

        base_dir = config.base_dir
        seq_num = '00'
        dataset = pykitti.odometry(base_dir, seq_num)

        # take a sample point cloud
        num_poses = len(dataset.poses)
        np.random.seed(0)
        sample_n = 1
        sample_idx = np.random.randint(0, num_poses, sample_n)
        sample_velo = dataset.get_velo(sample_idx[0])

        # create a random transformation
        max_angle = 5.0
        anglex = np.random.uniform(-1.0, 1.0) * np.pi / 180.0
        angley = np.random.uniform(-1.0, 1.0) * np.pi / 180.0
        anglez = max_angle * np.random.uniform(-1.0, 1.0) * np.pi / 180.0

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                       [0, cosx, -sinx],
                       [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                       [0, 1, 0],
                       [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-1, 2), np.random.uniform(0.2, 0.2),
                                   np.random.uniform(-0.2, 0.2)])
        rand_T = np.eye(4)
        rand_T[:3,:3] = R_ab
        rand_T[:3, 3] = translation_ab


        # convert to image
        velo_img_orig, _ = pc2img(sample_velo.copy(), config, rand_T=None, debug=True)
        velo_img, velo_range = pc2img(sample_velo.copy(), config, rand_T=rand_T, debug=True)
        velo_img_slow = pc2img_slow(sample_velo.copy(), config, rand_T=rand_T)

        # verify these two produce same results
        diff = velo_img - velo_img_slow
        anomaly = np.count_nonzero(diff)
        if anomaly == 0:
            print('======================================')
            print('======UNIT TEST PC2IMG SUCCESS========')
            print('======================================')