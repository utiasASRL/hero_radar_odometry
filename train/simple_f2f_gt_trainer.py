import signal
import os
from os import makedirs, remove
from os.path import exists, join
import time

# Dataset
from datasets.custom_sampler import *
from datasets.kitti import *
from utils.config import *
from torch.utils.data import DataLoader
import torch.nn.functional as F

# network
from models.f2f_gt_net import F2FGTNet

# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    # Initialize datasets
    config = KittiConfig()
    training_dataset = KittiDataset(config, set='training')

    # Initialize sampler
    training_sampler = RandomWindowBatchSampler(batch_size=config.batch_size,
                                                window_size=config.window_size,
                                                seq_len=training_dataset.seq_len,
                                                drop_last=True)

    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_sampler=training_sampler,
                                 num_workers=config.num_workers,
                                 pin_memory=True)

    # gpu
    device = torch.device("cuda:0")

    # network
    net = F2FGTNet(config)
    net.to(device)

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)

    # Path of the result folder
    if config.saving:
        # if config.saving_path is None:
        config.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
        if not exists(config.saving_path):
            makedirs(config.saving_path)
        checkpoint_directory = join(config.saving_path, 'checkpoints')
        if not exists(checkpoint_directory):
            makedirs(checkpoint_directory)

    # epoch
    max_epoch = 10
    epoch_step = 500
    for epoch in range(max_epoch):
        net.train()
        for i_batch, sample_batch in enumerate(training_loader):
            # zero optimizer gradients
            optimizer.zero_grad()

            # vertex = sample_batch['vertex']
            # move to GPUvertex[3:, :, :]
            vertex = sample_batch['image'][:, :3, :, :]  # xyz
            vertex = vertex.to(device)
            T_iv = sample_batch['T_iv'].to(device)

            # forward pass
            keypoints_3D, keypoints_scores, pseudo_ref = net(vertex, T_iv)

            # loss
            loss = net.loss(keypoints_3D, keypoints_scores, T_iv, pseudo_ref)

            # backward pass
            loss.backward()

            # update
            optimizer.step()

            # console output
            print(epoch, i_batch, loss.item())

            # Log file
            if config.saving:
                with open(join(config.saving_path, 'training.txt'), "a") as file:
                    message = '{:d} {:d} {:.3f}\n'
                    file.write(message.format(epoch, i_batch, loss.item()))

                # also save every 500 steps
                if np.mod(i_batch, epoch_step) == 0:
                    # Get current state dict
                    save_dict = {'model_state_dict': net.state_dict(),
                                 'optimizer_state_dict': optimizer.state_dict(),
                                 'saving_path': config.saving_path}

                    # Save current state of the network (for restoring purposes)
                    checkpoint_path = join(checkpoint_directory, 'current_chkp.tar')
                    torch.save(save_dict, checkpoint_path)


        # save every epoch
        if config.saving:
            # Get current state dict
            save_dict = {'model_state_dict': net.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'saving_path': config.saving_path}

            # Save current state of the network (for restoring purposes)
            checkpoint_path = join(checkpoint_directory, 'current_chkp.tar')
            torch.save(save_dict, checkpoint_path)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)