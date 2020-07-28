#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Configuration class
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


from os.path import join
import numpy as np


# Colors for printing
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class KittiConfig:
    """
    Class containing the parameters you want to modify for this dataset
    """

    ##################
    # Input parameters
    ##################

    # dataset base directory
    # base_dir = '/mnt/ssd1/research/dataset/KITTI/dataset'
    base_dir = '/home/david/Data/kitti'

    # training set seq
    # train_seq = ['{:02d}'.format(i) for i in range(11) if (i != 8 and i != 9)]
    train_seq = ['{:02d}'.format(i) for i in range(11) if i == 0]
    val_seq = ['{:02d}'.format(i) for i in range(11) if i == 8]
    test_seq = ['{:02d}'.format(i) for i in range(11, 22) if i == 9]

    # azimuth resolution in deg
    azi_res = 0.4375

    # min and max for azimuth in deg
    azi_min = -180.0
    azi_max = 180.0

    # elevation resolution in deg
    ele_res = 0.4375

    # min and max for elevation in deg
    ele_min = -25.0
    ele_max = 3.0

    # input features
    # 0: intensity + range
    # 1: xyz
    # 2: intensity + xyz
    # 3: all
    in_feat_setting = 1
    in_feat = 3

    # window size
    window_size = 2

    # batch size
    batch_size = 3

    # training
    learning_rate = 1e-4
    num_workers = 2

    # saving
    saving = True

    def __init__(self):
        """
        Class Initializer
        """

        self.saving_path = ''

        pass

    def load(self, path):

        pass

    def save(self):

        with open(join(self.saving_path, 'parameters.txt'), "w") as text_file:

            text_file.write('# -----------------------------------#\n')

