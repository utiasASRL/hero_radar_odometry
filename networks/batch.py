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
from utils.time_utils import sync_ntp_to_ros_time, get_ros_gps_times, get_gps_time_from_utc_stamp

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
        self.lidar_z_lim = config['plot']['lidar_z_lim']
        self.plot_dir = config['plot']['dir']
        self.frame_counter = 0

        # set solver parameters
        self.solver.setExtrinsicPriorCov(np.array(config['steam']['ex_prior_var']).reshape(6, 1))
        self.solver.setRadarCov(np.array(config['steam']['radar_var']).reshape(3, 1))
        self.solver.setLidarCov(np.array(config['steam']['lidar_var']).reshape(6, 1))
        self.solver.setRadarRobustCost(config['steam']['robust_radar_cost'])

        # determine directory with lidar data
        temp = get_sequences(config['data_dir'], '')
        root = os.path.join(config['data_dir'], temp[config['test_split'][0]])
        self.lidar_dir = os.path.join(root, 'lidar')
        self.lidar_frames = get_frames(self.lidar_dir, extension='.bin')

        # TODO
        # self.time_offset = int((9599351 - 2e5)*1e3)
        self.time_offset = 0
        applanix_time_file = os.path.join(root, 'applanix/ros_and_gps_time.csv')
        self.appl_ros_times, self.appl_gps_times = get_ros_gps_times(applanix_time_file)
        self.sync_params = sync_ntp_to_ros_time(root)

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

        self.frame_counter += 1
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

        # convert times1 and times2 to UTC?
        times1 = (self.convert_ntp_to_utc(times1)*1e6).astype(np.int64)
        times2 = (self.convert_ntp_to_utc(times2)*1e6).astype(np.int64)

        timestamps1 = getApproxTimeStamps([points1], [times1])[0]
        timestamps2 = getApproxTimeStamps([points2], [times2])[0]

        # add measurements
        self.solver.addFramePair(points2, points1,
            timestamps2, timestamps1, np.min(timestamps1), np.max(timestamps2))

        if save_for_plotting:
            timestamps1 = getApproxTimeStamps([tgt1], [times1])[0]
            self.radar_frames += [{'points': tgt1, 'times': timestamps1, 'frame_id': self.frame_counter}]

        return

    def convert_ntp_to_utc(self, ntptimes):
        # ntptime = ntptimes[0] / 1.0e6
        # rostime = ntptime
        # if self.sync_params is not None:
        #     rostime = ntptime - (self.sync_params[0] + self.sync_params[1] * ntptime)
        # gpstime = get_gps_time_from_utc_stamp(rostime, self.appl_ros_times, self.appl_gps_times)
        return ntptimes*1e-6 - (self.sync_params[0] + self.sync_params[1] * ntptimes*1e-6)


    def find_closest_lidar_frame(self, start_time, end_time):
        for i, file in enumerate(self.lidar_frames):
            lidar_time = (int(file[:-4]) + self.time_offset)*1e-3  # last 4 characters should be '.bin'
            if start_time < lidar_time < end_time:
                return True, i, file, lidar_time*1e3    # found frame

        # did not find appropriate frame
        return False, -1, 'N/A', 0

    def plot_frames(self):
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        for i in range(len(self.radar_frames)):
            self.plot_frame(i)

    def plot_frame(self, i):
        # radar
        radar_points = self.radar_frames[i]['points']
        radar_times = self.radar_frames[i]['times']
        frame_id = self.radar_frames[i]['frame_id']

        # get corresponding lidar frame
        success, _, filename, lidar_start_time = self.find_closest_lidar_frame(radar_times.min(), radar_times.max())
        lidar_data = np.fromfile(os.path.join(self.lidar_dir, filename), dtype=np.float32).reshape(-1, 6)

        # filter by height to remove ground
        ids = np.nonzero((self.lidar_z_lim[0] < lidar_data[:, 2])*(lidar_data[:, 2] < self.lidar_z_lim[1]))
        lidar_points = lidar_data[ids[0], :3]
        lidar_times = lidar_data[ids[0], 5].astype(np.float64)

        # correct lidar times (TODO: need to account for wrap around 1 hour)
        start_hour_sec = (lidar_start_time - int(lidar_start_time % (3600 * 1e9))) / 1e9
        lidar_times = ((lidar_times + start_hour_sec)*1e6).astype(np.int64)    # microseconds

        # also plot using known transform from other calibration method
        radar_points_known = radar_points.copy()
        T_rl = np.array([[6.861993198242921643e-01, 7.274135642622281406e-01, 0.0, 0.0],
                         [7.274135642622281406e-01, -6.861993198242921643e-01, 0.0, 0.0],
                         [0.0, 0.0, -1.0, 0.21],
                         [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        # undistort and transform to vehicle frame
        self.solver.undistortPointcloud(radar_points, radar_times, int(lidar_start_time*1e-3), T_rl, 0)
        self.solver.undistortPointcloud(radar_points_known, radar_times, int(lidar_start_time*1e-3), T_rl, 1)
        self.solver.undistortPointcloud(lidar_points, lidar_times, int(lidar_start_time*1e-3), T_rl, 2)

        plt.figure(figsize=(12, 12))
        plt.plot(lidar_points[::5, 0], lidar_points[::5, 1], '.r',
                 label='lidar z = (' + str(self.lidar_z_lim[0]) + 'm, ' + str(self.lidar_z_lim[1]) + 'm)')
        plt.plot(radar_points_known[:, 0], radar_points_known[:, 1], '.g', label='radar (given)')
        plt.plot(radar_points[:, 0], radar_points[:, 1], '.b', label='radar (estimated)')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')
        plt.xlim((-75, 75))
        plt.ylim((-75, 75))
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.plot_dir, 'frame_' + str(frame_id) + '.pdf'), bbox_inches='tight')
        plt.close()
