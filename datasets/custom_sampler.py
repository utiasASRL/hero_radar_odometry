from itertools import accumulate
from torch.utils.data import Sampler, SubsetRandomSampler

class RandomWindowBatchSampler(Sampler):
    def __init__(self, batch_size, window_size, seq_lens, drop_last=True, speed_filter=0.0, frame_speeds=None):
        valid_frames = []
        seq_len_cumsum = list(accumulate([0] + seq_lens))
        for j, len_j in enumerate(seq_len_cumsum[1:]):
            if speed_filter > 0.0 and frame_speeds is not None:
                valid_frame_temp = list(range(seq_len_cumsum[j], len_j - window_size + 1))
                valid_frame_temp2 = []
                for frame_idx in valid_frame_temp:
                    if frame_speeds[frame_idx] > speed_filter:
                        valid_frame_temp2.append(frame_idx)
                valid_frames += valid_frame_temp2
            else:
                valid_frames += list(range(seq_len_cumsum[j], len_j - window_size + 1))
        self.sampler = SubsetRandomSampler(valid_frames)
        self.batch_size = batch_size
        self.window_size = window_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch += [idx + k for k in range(self.window_size)]
            if len(batch) == self.batch_size*self.window_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        total_size = self.batch_size*self.window_size
        if self.drop_last:
            return len(self.sampler) // total_size
        return (len(self.sampler) + total_size - 1) // total_size

class SequentialWindowBatchSampler(Sampler):
    def __init__(self, batch_size, window_size, seq_lens, drop_last=True):
        self.valid_frames = []
        seq_len_cumsum = list(accumulate([0] + seq_lens))
        for j, len_j in enumerate(seq_len_cumsum[1:]):
            self.valid_frames += list(range(seq_len_cumsum[j], len_j - window_size + 1))
        self.batch_size = batch_size
        self.window_size = window_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in range(len(self.valid_frames)):
            batch += [idx + k for k in range(self.window_size)]
            if len(batch) == self.batch_size*self.window_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        total_size = self.batch_size*self.window_size
        if self.drop_last:
            return len(self.valid_frames) // total_size
        return (len(self.valid_frames) + total_size - 1) // total_size
