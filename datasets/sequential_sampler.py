from torch.utils.data import Sampler, SubsetRandomSampler, SequentialSampler
from itertools import accumulate

class SequentialWindowBatchSampler(Sampler):
    """Wraps another sampler to yield a mini-batch of windowed indices.
    Args:
        batch_size (int): Size of mini-batch.
        window_size (int): Size of each window.
        valid_frames (list) List of valid window frames.
    """

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
        else:
            return (len(self.valid_frames) + total_size - 1) // total_size
