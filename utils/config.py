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


class Config:
    """
    Class containing the parameters you want to modify for this dataset
    """

    ##################
    # Input parameters
    ##################

    # dataset base directory
    base_dir = '/mnt/ssd1/research/dataset/KITTI/dataset'

    # azimuth resolution in deg
    azi_res = 0.5

    # min and max for azimuth in deg
    azi_min = -180.0
    azi_max = 180.0

    # elevation resolution in deg
    ele_res = 0.5

    # min and max for elevation in deg
    ele_min = -25.0
    ele_max = 5.0

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

