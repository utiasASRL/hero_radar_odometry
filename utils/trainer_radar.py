import os
import sys
from time import time
import torch
import numpy as np
from tensorboardX import SummaryWriter

from utils.early_stopping import EarlyStopping
from utils.lie_algebra import so3_to_rpy
from utils.plot import plot_epoch_losses, plot_epoch_errors

class Monitor():
    def __init__(self, model, log_dir, valid_loader, gpuid, config):
        self.model = model
        self.log_dir = log_dir
        self.valid_loader = valid_loader
        self.gpuid = gpuid
        self.config = config
        self.dt = 0
        self.current_time = 0
        self.counter = 0
        print('Monitor running and saving to {}'.format(log_dir))
        self.writer = SummaryWriter(log_dir)

    def step(self):
        asdf

    def vis(self):
        batch_list = [bi, batch in enumerate(self.valid_loader)]
        ixes = np.linspace(0, len(batch_list) - 1, config['vis_num']).astype(np.int32)
        batch_list = [batch_list[ix] for ix in ixes]
        for bi, batch in batch_list:
            # TODO: run the model to visualize the output
            # Visualize the detector scores
            # Visualize the point matching quality
            loss = self.model()


class Trainer():

    def __init__(self, model, train_loader, valid_loader, config, pretrain_path=None):
        self.model = model
        self.model.to(config['gpuid'])
        if pretrain_path is not None:
            self.resume_checkpoint(pretrain_path)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.config = config

    def train_epoch(self, epoch):

        self.model.train()

        for i_batch, batch_sample in enumerate(self.train_loader):

            self.optimizer.zero_grad()
            loss = self.model(batch_sample, epoch)
            if loss.requires_grad:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_norm'])
            self.optimizer.step()

            # record number of inliers
            num_inliers, num_nonzero_weights = self.model.get_inliers(epoch)
            total_inliers += num_inliers
            total_nonzero_weights += num_nonzero_weights

            # record prediction error for each DOF
            if epoch >= self.config['loss']['start_svd_epoch']:
                total_sq_error += torch.sum(self.model.get_pose_error()**2, dim=0).unsqueeze(0).detach().cpu()

            # record loss
            for key in loss:
                if key in total_loss:
                    total_loss[key] += loss[key].item()
                else:
                    total_loss[key] = loss[key].item()

            # Console print (only one per second)
            if (t[-1] - last_display) > 60.0:
                last_display = t[-1]
                self.model.print_loss(loss, epoch, i_batch)
                self.model.print_inliers(epoch, i_batch)

            count += 1

        # Get and print summary statistics for the epoch
        print('\nTraining epoch: {}'.format(epoch))

        avg_inliers = total_inliers / float(count * self.config['train_loader']['batch_size'])
        avg_nonzero_weights = total_nonzero_weights / float(count * self.config['train_loader']['batch_size'])

        for key in total_loss:
            avg_loss = total_loss[key] / count
            if key in self.epoch_loss_train:
                self.epoch_loss_train[key].append(avg_loss)
            else:
                self.epoch_loss_train[key] = [avg_loss]
            print('{}: {:.6f}'.format(key, avg_loss))

        num_samples = count * self.config['train_loader']['batch_size']
        rms_error = torch.sqrt(total_sq_error / num_samples)
        if self.epoch_error_train is None:
            self.epoch_error_train = rms_error
        else:
            self.epoch_error_train = np.concatenate((self.epoch_error_train, rms_error), axis=0)

        print('Avg error:          {}'.format(rms_error))
        print('Avg inliers:        {}'.format(avg_inliers))
        print('Avg nonzero weight: {}'.format(avg_nonzero_weights))
        print('Time:               {}'.format(time.time() - start))
        print('\n')

        return self.epoch_loss_train['LOSS'][-1]

    def valid_epoch(self, epoch):
        """
        Validation logic for an epoch
        :param epoch: Integer, current training epoch.
        """
        total_loss = {}
        total_sq_error = torch.zeros(1,6)
        total_inliers = 0
        total_nonzero_weights = 0

        self.model.eval()

        t = [time.time()]
        last_display = time.time()
        start = time.time()

        count = 0
        with torch.no_grad():

                for i_batch, batch_sample in enumerate(self.valid_loader):

                    try:
                        loss = self.model(batch_sample, epoch)
                    except Exception as e:
                        self.model.print_loss(loss, epoch, i_batch)
                        self.model.print_inliers(epoch, i_batch)
                        print(e)
                        continue

                    t += [time.time()]

                    # record number of inliers
                    num_inliers, num_nonzero_weights = self.model.get_inliers(epoch)
                    total_inliers += num_inliers
                    total_nonzero_weights += num_nonzero_weights

                    # record prediction error for each DOF
                    if epoch >= self.config['loss']['start_svd_epoch']:
                        total_sq_error += torch.sum(self.model.get_pose_error()**2, dim=0).unsqueeze(0).detach().cpu()

                    # record loss
                    for key in loss:
                        if key in total_loss:
                            total_loss[key] += loss[key].item()
                        else:
                            total_loss[key] = loss[key].item()

                    # Console print (only one per minute)
                    if (t[-1] - last_display) > 60.0:
                        last_display = t[-1]
                        self.model.print_loss(loss, epoch, i_batch)
                        self.model.print_inliers(epoch, i_batch)

                    count += 1

        # Get and print summary statistics for the epoch
        print('\nValidation epoch: {}'.format(epoch))

        avg_inliers = total_inliers / float(count * self.config['train_loader']['batch_size'])
        avg_nonzero_weights = total_nonzero_weights / float(count * self.config['train_loader']['batch_size'])

        for key in total_loss:
            avg_loss = total_loss[key] / count
            if key in self.epoch_loss_valid:
                self.epoch_loss_valid[key].append(avg_loss)
            else:
                self.epoch_loss_valid[key] = [avg_loss]
            print('{}: {:.6f}'.format(key, avg_loss))

        num_samples = count * self.config['train_loader']['batch_size']
        rms_error = torch.sqrt(total_sq_error / num_samples)
        if self.epoch_error_valid is None:
            self.epoch_error_valid = rms_error
        else:
            self.epoch_error_valid = np.concatenate((self.epoch_error_valid, rms_error), axis=0)

        print('Avg error:          {}'.format(rms_error))
        print('Avg inliers:        {}'.format(avg_inliers))
        print('Avg nonzero weight: {}'.format(avg_nonzero_weights))
        print('Time:               {}'.format(time.time() - start))
        print('\n')

        return self.epoch_loss_valid['LOSS'][-1]

    def train(self):
        for epoch in range(self.start_epoch, self.config['max_epochs']):
            train_loss = self.train_epoch(epoch)
            torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()}, self.checkpoint_path)

    def resume_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        print("Resume training from epoch {}".format(self.start_epoch))
