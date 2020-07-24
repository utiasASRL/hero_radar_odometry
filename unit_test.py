# sys imports
import os
import sys

# third-party imports
import pykitti
import numpy as np
from PIL import Image

# project imports
from utils.config import Config
from utils.helper_func import pc2img

# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':
    test_category = {
        'PC2IMG': True
    }

    ##########################
    # UNIT TEST ON PC2IMG FUNC
    ##########################
    if test_category['PC2IMG']:
        config = Config()

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
        velo_img, velo_range = pc2img(sample_velo, config)

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