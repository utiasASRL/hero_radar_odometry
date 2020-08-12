#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling SemanticKitti dataset.
#      Implements a Dataset, a Sampler, and a collate_fn
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# sys imports
import os
import time
from os import listdir
from os.path import exists, join, isdir

# third-party imports
import numpy as np
import torch
from torch.utils.data import Dataset

# project imports
from utils.helper_func import load_lidar_image


# ----------------------------------------------------------------------------------------------------------------------
#
#           Dataset class definition
#       \******************************/


class KittiDataset(Dataset):
    """Class to handle Kitti dataset."""

    def __init__(self, config, set='training'):
        Dataset.__init__(self)

        ##########################
        # Parameters for the files
        ##########################

        # Dataset folder
        self.path = config["dataset"]["data_dir"]

        # Training or test set
        self.set = set

        # Get a list of sequences
        if self.set == 'training':
            self.sequences = config["dataset"]["seq"]["train"]
        elif self.set == 'validation':
            self.sequences = config["dataset"]["seq"]["val"]
        elif self.set == 'test':
            self.sequences = config["dataset"]["seq"]["test"]
        else:
            raise ValueError('Unknown set for SemanticKitti data: ', self.set)

        # List all files in each sequence
        self.frames = []
        for seq in self.sequences:
            velo_path = join(self.path, 'sequences', seq, 'velodyne')
            frames = np.sort([vf[:-4] for vf in listdir(velo_path) if vf.endswith('.bin')])

            # # UNCOMMENT these lines to only use certain segment of a sequence
            # num_frames = 200
            # frames = frames[:num_frames]

            self.frames.append(frames)

        ##################
        # Other parameters
        ##################

        # Store height and width of image # TODO make this a variable but not fixed
        self.height = config["dataset"]["images"]["height"]
        self.width = config["dataset"]["images"]["width"]

        # Parameters from config
        self.config = config

        ##################
        # Load calibration
        ##################

        # Init variables
        self.calibrations = []
        self.times = []
        self.poses = []
        self.all_inds = None
        self.val_confs = []
        self.seq_len = [len(_) for _ in self.frames]

        # Load everything
        self.load_calib_poses()

        return

    def __len__(self):
        """
        Return the length of data here
        """
        return self.all_inds.shape[0]

    def __getitem__(self, i):

        s_ind, f_ind = self.all_inds[i]

        # Get center of the first frame in world coordinates
        T_iv = self.poses[s_ind][f_ind]

        # Path of points and labels
        seq_path = join(self.path, 'sequences', self.sequences[s_ind])
        velo_file = join(seq_path, 'velodyne', self.frames[s_ind][f_ind] + '.bin')

        # Read points
        points = np.fromfile(velo_file, dtype=np.float32)
        points = points.reshape((-1, 4))

        # convert to image
        geometry_img, input_img = load_lidar_image(points, self.config, debug=False)

        # store height and width
        self.height, self.width, _ = geometry_img.shape

        return {'geometry': geometry_img, 'input': input_img, 's_ind': s_ind, 'f_ind': f_ind, 'T_iv': T_iv}

    def load_calib_poses(self):
        """
        load calib poses and times.
        """

        ###########
        # Load data
        ###########

        self.calibrations = []
        self.times = []
        self.poses = []

        for seq in self.sequences:

            seq_folder = join(self.path, 'sequences', seq)

            # Read Calib
            self.calibrations.append(self.parse_calibration(join(seq_folder, "calib.txt")))

            # Read times
            self.times.append(np.loadtxt(join(seq_folder, 'times.txt'), dtype=np.float32))

            # Read poses
            poses_f64 = self.parse_poses(join(seq_folder, 'poses.txt'), self.calibrations[-1])
            self.poses.append([pose.astype(np.float32) for pose in poses_f64])

        ###################################
        # Prepare the indices of all frames
        ###################################

        seq_inds = np.hstack([np.ones(len(_), dtype=np.int32) * i for i, _ in enumerate(self.frames)])
        frame_inds = np.hstack([np.arange(len(_), dtype=np.int32) for _ in self.frames])
        self.all_inds = np.vstack((seq_inds, frame_inds)).T

        return

    def parse_calibration(self, filename):
        """ read calibration file with given filename

            Returns
            -------
            dict
                Calibration matrices as 4x4 numpy arrays.
        """
        calib = {}

        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()

        return calib

    def parse_poses(self, filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(pose, Tr))

        return poses