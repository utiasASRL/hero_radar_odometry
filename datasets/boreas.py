"""
    PyTorch dataset class for The University of Toronto Boreas Dataset.
    Authors: Keenan Burnett
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets.custom_sampler import RandomWindowBatchSampler, SequentialWindowBatchSampler
from datasets.radar import load_radar, radar_polar_to_cartesian
from utils.utils import get_inverse_tf, get_transform
from datasets.oxford import OxfordDataset, mean_intensity_mask

def roll(r):
    return np.array([[1, 0, 0], [0, np.cos(r), np.sin(r)], [0, -np.sin(r), np.cos(r)]], dtype=np.float64)

def pitch(p):
    return np.array([[np.cos(p), 0, -np.sin(p)], [0, 1, 0], [np.sin(p), 0, np.cos(p)]], dtype=np.float64)

def yaw(y):
    return np.array([[np.cos(y), np.sin(y), 0], [-np.sin(y), np.cos(y), 0], [0, 0, 1]], dtype=np.float64)

def yawPitchRollToRot(y, p, r):
    Y = yaw(y)
    P = pitch(p)
    R = roll(r)
    C = np.matmul(P, Y)
    return np.matmul(R, C)

def rotToYawPitchRoll(C, eps = 1e-15):
    i = 2
    j = 1
    k = 0
    c_y = np.sqrt(C[i, i]**2 + C[j, i]**2)
    if c_y > eps:
        r = np.arctan2(C[j, i], C[i, i])
        p = np.arctan2(-C[k, i], c_y)
        y = np.arctan2(C[k, j], C[k, k])
    else:
        r = 0
        p = np.arctan2(-C[k, i], c_y)
        y = np.arctan2(-C[j, k], C[j, j])
    return y, p, r

def get_transform_boreas(gt):
    # gt: list of floats or doubles
    T = np.identity(4, dtype=np.float64)
    C_enu_sensor = yawPitchRollToRot(gt[10], gt[9], gt[8])
    T[0, 3] = gt[2]
    T[1, 3] = gt[3]
    T[2, 3] = gt[4]
    T[0:3, 0:3] = C_enu_sensor
    return T

class BoreasDataset(OxfordDataset):
    """Boreas Radar Dataset"""
    dataset_prefix = 'boreas'

    def get_frames_with_gt(self, frames, gt_path):
        # For the Boreas dataset, this operation is not needed but is preserved for backwards compatibility
        return frames

    def get_groundtruth_odometry(self, radar_time, gt_path):
        """For a given time stamp (UNIX INT64), returns 4x4 transformation matrix from current time to next."""
        def parse(gps_line):
            out = [float(x) for x in gps_line.split(',')]
            out[0] = int(gps_line.split(',')[0])
            return out
        with open(gt_path, 'r') as f:
            f.readline()
            lines = f.readlines()
            for i in range(len(lines) - 1):
                gt1 = parse(lines[i])
                if gt1[0] == radar_time:
                    gt2 = parse(lines[i + 1])
                    T_enu_r1 = get_transform_boreas(gt1)
                    T_enu_r2 = get_transform_boreas(gt2)
                    T_r2_r1 = np.matmul(get_inverse_tf(T_enu_r2), T_enu_r1)  # 4x4 SE(3)
                    heading, _, _ = rotToYawPitchRoll(T_r2_r1[0:3, 0:3])
                    T_2_1 = get_transform(T_r2_r1[0, 3], T_r2_r1[1, 3], heading)  # 4x4 SE(2)
                    return T_2_1
        assert(0), 'ground truth transform for {} not found in {}'.format(radar_time, gt_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        seq = self.get_seq_from_idx(idx)
        frame = self.data_dir + seq + '/radar/' + self.frames[idx]
        _, azimuths, _, polar, _ = load_radar(frame, navtech_version=CIR204)
        data = radar_polar_to_cartesian(azimuths, polar, self.config['radar_resolution'],
                                        self.config['cart_resolution'], self.config['cart_pixel_width'],
                                        navtech_version=CIR204)  # 1 x H x W
        polar_mask = mean_intensity_mask(polar, self.mean_int_mask_mult)
        mask = radar_polar_to_cartesian(azimuths, polar_mask, self.config['radar_resolution'],
                                        self.config['cart_resolution'], self.config['cart_pixel_width'],
                                        navtech_version=CIR204).astype(np.float32)
        # Get ground truth transform between this frame and the next
        time1 = int(self.frames[idx].split('.')[0])
        if idx + 1 < len(self.frames):
            time2 = int(self.frames[idx + 1].split('.')[0])
        else:
            time2 = 0
        times = np.array([time1, time2]).reshape(1, 2)
        T_21 = self.get_groundtruth_odometry(time1, self.data_dir + seq + '/applanix/radar_poses.csv')
        return {'data': data, 'T_21': T_21, 'times': times, 'mask': mask}

def get_dataloaders_boreas(config):
    """Retrieves train, validation, and test data loaders."""
    vconfig = dict(config)
    vconfig['batch_size'] = 1
    train_dataset = BoreasDataset(config, 'train')
    valid_dataset = BoreasDataset(vconfig, 'validation')
    test_dataset = BoreasDataset(vconfig, 'test')
    train_sampler = RandomWindowBatchSampler(config['batch_size'], config['window_size'], train_dataset.seq_lens)
    valid_sampler = SequentialWindowBatchSampler(1, config['window_size'], valid_dataset.seq_lens)
    test_sampler = SequentialWindowBatchSampler(1, config['window_size'], test_dataset.seq_lens)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=config['num_workers'])
    valid_loader = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=config['num_workers'])
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=config['num_workers'])
    return train_loader, valid_loader, test_loader
