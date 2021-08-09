import os
import torch
import numpy as np
from utils.utils import convert_to_radar_frame, mask_intensity_filter
import cv2
import cpp.build.BatchSolver as steamcpp
from datasets.features import *
from utils.utils import getApproxTimeStamps

import matplotlib.pyplot as plt
from datasets.oxford import get_sequences, get_frames

class Batch():
    """Hand-crafted feature extraction (cen2018) and descriptors (orb)."""

    def __init__(self, config):
        self.config = config
        self.orb = cv2.ORB_create()
        self.orb.setPatchSize(21)
        self.orb.setEdgeThreshold(21)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING) #, crossCheck=True)
        self.radar_resolution = config['radar_resolution']
        self.cart_resolution = config['cart_resolution']
        self.cart_pixel_width = config['cart_pixel_width']
        self.solver = steamcpp.BatchSolver()
        self.radar_frames = [] # for plotting

        # determine directory with lidar data
        temp = get_sequences(config['data_dir'], '')
        self.lidar_dir = os.path.join(config['data_dir'], temp[config['test_split'][0]], 'lidar')
        self.lidar_frames = get_frames(self.lidar_dir, extension='.bin')

        # TODO
        self.time_offset = int((9599351 - 2e5)*1e3)
        # self.time_offset = 0

    def add_frame_pair(self, batch, save_for_plotting=False):
        data = batch['data']
        timestamps = batch['timestamps']
        # t_ref = batch['t_ref']
        tgt_ids = [1]
        src_ids = [0]
        polar = batch['polar'].numpy().squeeze()
        azimuths = batch['azimuths'].numpy().squeeze()
        ptgt1 = cen2018features(polar[0])
        ptgt2 = cen2018features(polar[1])
        tgt1 = polar_to_cartesian_points(azimuths[0], ptgt1, self.radar_resolution)
        tgt2 = polar_to_cartesian_points(azimuths[1], ptgt2, self.radar_resolution)
        tgt1, tgt1p = convert_to_bev2(tgt1, self.cart_resolution, self.cart_pixel_width, 21)
        tgt2, tgt2p = convert_to_bev2(tgt2, self.cart_resolution, self.cart_pixel_width, 21)
        kp1 = convert_to_keypoints(tgt1p)
        kp2 = convert_to_keypoints(tgt2p)
        img1 = (data[0] * 255).numpy().squeeze().astype(np.uint8)
        img2 = (data[1] * 255).numpy().squeeze().astype(np.uint8)
        kp1_, des1 = self.orb.compute(img1, kp1)
        kp2_, des2 = self.orb.compute(img2, kp2)
        assert(len(kp1_) == len(kp1))

        matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        N = len(good_matches)

        if N == 0:
            print("Warning: No good matches, skipping this frame")
            return

        p1 = np.zeros((N, 2))
        p2 = np.zeros((N, 2))
        p1p = np.zeros((N, 2))
        p2p = np.zeros((N, 2))
        for i,match in enumerate(good_matches):
            p1[i, :] = tgt1[match.queryIdx, :]
            p1p[i, :] = tgt1p[match.queryIdx, :]
            p2[i, :] = tgt2[match.trainIdx, :]
            p2p[i, :] = tgt2p[match.trainIdx, :]

        p1 = np.expand_dims(p1, axis=0)
        p2 = np.expand_dims(p2, axis=0)
        p1p = np.expand_dims(p1p, axis=0)
        p2p = np.expand_dims(p2p, axis=0)
        pseudo_coords_xy = torch.from_numpy(p1)
        keypoint_coords_xy = torch.from_numpy(p2)

        pseudo_coords = torch.from_numpy(p1p)
        keypoint_coords = torch.from_numpy(p2p)
        time_tgt = timestamps[tgt_ids]
        time_src = timestamps[src_ids]
        # t_ref_tgt = t_ref[tgt_ids]
        # t_ref_src = t_ref[src_ids]
        # keypoint_ints = torch.ones((1, 1, p1.shape[1]))
        # match_weights = torch.ones((1, 1, p1.shape[1]))

        # prepare input for cpp call
        points1 = pseudo_coords_xy[0].detach().cpu().numpy()
        points2 = keypoint_coords_xy[0].detach().cpu().numpy()
        times1 = time_src[0].cpu().numpy().squeeze()
        times2 = time_tgt[0].cpu().numpy().squeeze()
        timestamps1 = getApproxTimeStamps([points1], [times1])[0]
        timestamps2 = getApproxTimeStamps([points2], [times2])[0]

        # add measurements
        self.solver.addFramePair(points2, points1,
            timestamps2, timestamps1, np.min(timestamps1), np.max(timestamps2))

        if save_for_plotting:
            timestamps1 = getApproxTimeStamps([tgt1], [times1])[0]
            self.radar_frames += [{'points': tgt1, 'times': timestamps1}]

        # self.plot_frame(0)
        return

    def find_closest_lidar_frame(self, start_time, end_time):
        for i, file in enumerate(self.lidar_frames):
            lidar_time = (int(file[:-4]) + self.time_offset)*1e-3  # last 4 characters should be '.bin'
            if start_time < lidar_time < end_time:
                return True, i, file    # found frame

        # did not find appropriate frame
        return False, -1, 'N/A'

    def plot_frame(self, frame_num):
        # radar
        radar_points = self.radar_frames[frame_num]['points']
        radar_times = self.radar_frames[frame_num]['times']

        # get corresponding lidar frame
        success, frame_id, filename = self.find_closest_lidar_frame(radar_times.min(), radar_times.max())
        lidar_data = np.fromfile(os.path.join(self.lidar_dir, filename), dtype=np.float32).reshape(-1, 6)
        # lidar_data = np.fromfile(os.path.join(self.lidar_dir, self.lidar_frames[frame_id]), dtype=np.float32).reshape(-1, 6)
        ids = np.nonzero((-1 < lidar_data[:, 2])*(lidar_data[:, 2] < 1.5))
        lidar_points = lidar_data[ids[0], :3]
        lidar_times = lidar_data[ids[0], 5]
        # if x[2, i] < -1.5 or x[2, i] > 1.5:

        # C = np.array([[6.861993198242921643e-01, 7.274135642622281406e-01, 0.000000000000000000e+00],
        #               [7.274135642622281406e-01, -6.861993198242921643e-01, 0.000000000000000000e+00],
        #               [0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00]])
        # t = np.array([[0.0], [0.0], [0.21]])

        C = np.array([[0.669,     0.74,  -0.0624],
                      [0.741,   -0.671,  -0.0114],
                      [-0.0503,  -0.0386,   -0.998]])
        t = np.array([[0.0191], [-0.00912], [0.222]])

        # 0.635  0.728  0.259 -0.107
        # 0.741 -0.669 0.0646 -0.186
        #  0.22  0.151 -0.964  0.408
        #     0      0      0      1

        lidar_points = lidar_points@C.T + t.T

        # undistort and transform to vehicle frame
        # TODO

        plt.figure()
        plt.plot(lidar_points[::5, 0], lidar_points[::5, 1], '.r', label='lidar')

        # plt.axis('equal')
        #
        # plt.figure()
        plt.plot(radar_points[:, 0], radar_points[:, 1], '.b', label='radar')
        plt.axis('equal')

        plt.show()
        a = 1