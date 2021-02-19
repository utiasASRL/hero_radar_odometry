"""
    PyTorch dataset class for the Oxford Radar Robotcar Dataset.
    Authors: Keenan Burnett
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets.custom_sampler import RandomWindowBatchSampler, SequentialWindowBatchSampler
from datasets.radar import load_radar, radar_polar_to_cartesian
from utils.utils import get_inverse_tf, get_transform

def get_sequences(path, prefix='2019'):
    """Retrieves a list of all the sequences in the oxford dataset."""
    sequences = [f for f in os.listdir(path) if prefix in f]
    sequences.sort()
    return sequences

def get_frames(path, extension='.png'):
    """Retrieves all the file names within a path that match the given extension."""
    frames = [f for f in os.listdir(path) if extension in f]
    frames.sort()
    return frames

def check_if_frame_has_gt(frame, gt_lines):
    """Checks whether or not the specified file has ground truth within the corresponding radar_odometry.csv."""
    for i in range(len(gt_lines) - 1, -1, -1):
        line = gt_lines[i].split(',')
        if frame == int(line[9]):
            return True
    return False

def get_frame_speeds(frames, gt_path):
    frame_speeds = frames.copy()
    frame_angvels = frames.copy()
    with open(gt_path, 'r') as f:
        f.readline()
        lines = f.readlines()
        times = []
        speeds = []
        angvels = []
        for line in lines:
            line = line.split(',')
            times.append(int(line[9]))
            speeds.append(np.sqrt(float(line[2])**2 + float(line[3])**2) / 0.25)
            angvels.append(np.abs(float(line[7])) / 0.25)

        for i, frame in enumerate(frames):
            radar_time = int(frame.split('.')[0])
            v = None
            for j, time in enumerate(times):
                if time == radar_time:
                    v = speeds[j]
                    frame_speeds[i] = speeds[j]
                    frame_angvels[i] = angvels[j]
                    del times[j]
                    del speeds[j]
                    del angvels[j]
                    break
            assert(v is not None),"could not find a speed for idx: {}".format(i)
    return frame_speeds, frame_angvels

def get_frames_with_gt(frames, gt_path):
    """Returns a subset of the specified file list which have ground truth."""
    # For the Oxford Dataset we do a search from the end backwards because some
    # of the sequences don't have GT as the end, but they all have GT at the beginning.
    frames_out = frames
    with open(gt_path, 'r') as f:
        f.readline()
        lines = f.readlines()
        for i in range(len(frames) - 1, -1, -1):
            frame = int(frames[i].split('.')[0])
            if check_if_frame_has_gt(frame, lines):
                break
            frames_out.pop()
    return frames_out

def get_groundtruth_odometry(radar_time, gt_path):
    """For a given time stamp (UNIX INT64), returns 4x4 homogeneous transformation matrix fromt current time to next."""
    with open(gt_path, 'r') as f:
        f.readline()
        lines = f.readlines()
        for line in lines:
            line = line.split(',')
            if int(line[9]) == radar_time:
                T = get_transform(float(line[2]), float(line[3]), float(line[7]))  # from next time to current
                return get_inverse_tf(T)  # T_2_1 from current time step to the next time step
    assert(0), 'ground truth transform for {} not found in {}'.format(radar_time, gt_path)

def mean_intensity_mask(polar_data, multiplier=3.0):
    num_azimuths, range_bins = polar_data.shape
    mask = np.zeros((num_azimuths, range_bins))
    for i in range(num_azimuths):
        m = np.mean(polar_data[i, :])
        mask[i, :] = polar_data[i, :] > multiplier * m
    return mask

class OxfordDataset(Dataset):
    """Oxford Radar Robotcar Dataset."""
    def __init__(self, config, split='train'):
        self.config = config
        self.data_dir = config['data_dir']
        sequences = get_sequences(self.data_dir)
        self.sequences = self.get_sequences_split(sequences, split)
        self.seq_idx_range = {}
        self.frames = []
        self.frame_speeds = []
        self.frame_angvels = []
        self.seq_lens = []
        self.mean_int_mask_mult = config['mean_int_mask_mult']
        for seq in self.sequences:
            seq_frames = get_frames(self.data_dir + seq + '/radar/')
            seq_frames = get_frames_with_gt(seq_frames, self.data_dir + seq + '/gt/radar_odometry.csv')
            seq_frame_speeds, seq_frame_angs = get_frame_speeds(seq_frames, self.data_dir + seq + '/gt/radar_odometry.csv')
            self.seq_idx_range[seq] = [len(self.frames), len(self.frames) + len(seq_frames)]
            self.seq_lens.append(len(seq_frames))
            self.frames.extend(seq_frames)
            self.frame_speeds.extend(seq_frame_speeds)
            self.frame_angvels.extend(seq_frame_angs)

    def get_sequences_split(self, sequences, split):
        """Retrieves a list of sequence names depending on train/validation/test split."""
        self.split = self.config['train_split']
        if split == 'validation':
            self.split = self.config['validation_split']
        elif split == 'test':
            self.split = self.config['test_split']
        return [seq for i, seq in enumerate(sequences) if i in self.split]

    def __len__(self):
        return len(self.frames)

    def get_seq_from_idx(self, idx):
        """Returns the name of the sequence that this idx belongs to."""
        for seq in self.sequences:
            if self.seq_idx_range[seq][0] <= idx and idx < self.seq_idx_range[seq][1]:
                return seq
        assert(0), 'sequence for idx {} not found in {}'.format(idx, self.seq_idx_range)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        seq = self.get_seq_from_idx(idx)
        frame = self.data_dir + seq + '/radar/' + self.frames[idx]
        _, azimuths, _, polar, _ = load_radar(frame)
        cart = radar_polar_to_cartesian(azimuths, polar, self.config['radar_resolution'],
                                        self.config['cart_resolution'], self.config['cart_pixel_width'])  # 1 x H x W
        #data = (cart - cart.mean()) / (cart.std() + 1e-8)
        data = cart
        polar_mask = mean_intensity_mask(polar, self.mean_int_mask_mult)
        mask = radar_polar_to_cartesian(azimuths, polar_mask, self.config['radar_resolution'],
                                        self.config['cart_resolution'],
                                        self.config['cart_pixel_width']).astype(np.float32)
        #mask = (mask > 0.5).astype(np.float32)
        # Get ground truth transform between this frame and the next
        time1 = int(self.frames[idx].split('.')[0])
        if idx + 1 < len(self.frames):
            time2 = int(self.frames[idx + 1].split('.')[0])
        else:
            time2 = 0
        times = np.array([time1, time2]).reshape(1, 2)
        T_21 = get_groundtruth_odometry(time1, self.data_dir + seq + '/gt/radar_odometry.csv')
        return {'data': data, 'cart': cart, 'T_21': T_21, 'times': times, 'mask': mask}

def get_dataloaders(config):
    """Retrieves train, validation, and test data loaders."""
    vconfig = dict(config)
    vconfig['batch_size'] = 1
    train_dataset = OxfordDataset(config, 'train')
    valid_dataset = OxfordDataset(vconfig, 'validation')
    test_dataset = OxfordDataset(vconfig, 'test')
    train_sampler = RandomWindowBatchSampler(config['batch_size'], config['window_size'], train_dataset.seq_lens)
    valid_sampler = SequentialWindowBatchSampler(1, config['window_size'], valid_dataset.seq_lens)
    test_sampler = SequentialWindowBatchSampler(1, config['window_size'], test_dataset.seq_lens)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=config['num_workers'])
    valid_loader = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=config['num_workers'])
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=config['num_workers'])
    return train_loader, valid_loader, test_loader
