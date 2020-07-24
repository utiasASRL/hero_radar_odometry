from torch.utils.data import Sampler, SubsetRandomSampler

class RandomWindowBatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of windowed indices.
    Args:
        batch_size (int): Size of mini-batch.
        window_size (int): Size of each window.
        valid_frames (list) List of valid window frames.
    """

    def __init__(self, batch_size, window_size, valid_frames, drop_last):

        self.sampler = SubsetRandomSampler(valid_frames)
        self.batch_size = batch_size
        self.window_size = window_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            for k in range(self.window_size):
                batch.append(idx + k)
            if len(batch) == self.batch_size*self.window_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        total_size = self.batch_size*self.window_size
        if self.drop_last:
            return len(self.sampler) // total_size
        else:
            return (len(self.sampler) + total_size - 1) // total_size