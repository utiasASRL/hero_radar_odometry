from itertools import accumulate
from torch.utils.data import Sampler, SubsetRandomSampler

class RandomWindowBatchSampler(Sampler):
    def __init__(self, batch_size, window_size, seq_len, drop_last=True):
        valid_frames = []
        seq_len_cumsum = list(accumulate([0] + seq_len))
        for j, len_j in enumerate(seq_len_cumsum[1:]):
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
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        total_size = self.batch_size*self.window_size
        if self.drop_last:
            return len(self.sampler) // total_size
        return (len(self.sampler) + total_size - 1) // total_size

class SequentialWindowBatchSampler(Sampler):
    def __init__(self, batch_size, window_size, seq_len, drop_last=True):
        self.valid_frames = []
        seq_len_cumsum = list(accumulate([0] + seq_len))
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
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        total_size = self.batch_size*self.window_size
        if self.drop_last:
            return len(self.valid_frames) // total_size
        return (len(self.valid_frames) + total_size - 1) // total_size
