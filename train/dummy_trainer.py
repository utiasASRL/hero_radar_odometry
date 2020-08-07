import signal
import os

import argparse
import json

# Dataset
from datasets.custom_sampler import *
from datasets.kitti import *
from utils.config import *
from torch.utils.data import DataLoader

# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    # Initialize datasets
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str,
                        help='config file path (default: None)')

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    training_dataset = KittiDataset(config, set='training')

    # Initialize sampler
    training_sampler = RandomWindowBatchSampler(batch_size=4,
                                                window_size=3,
                                                seq_len=training_dataset.seq_len,
                                                drop_last=True)

    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_sampler=training_sampler,
                                 num_workers=0,
                                 pin_memory=True)

    for i_batch, sample_batched in enumerate(training_loader):
        print(sample_batched['geometry'].shape)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)