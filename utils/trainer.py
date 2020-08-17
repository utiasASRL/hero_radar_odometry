import os
import sys
import time
import torch
import numpy as np

from utils.early_stopping import EarlyStopping
from utils.lie_algebra import so3_to_rpy
from utils.plot import plot_epoch_losses, plot_epoch_errors

class Trainer():
    """
    Trainer class
    """
    def __init__(self, model, train_loader, valid_loader, config, result_path, session_path, checkpoint_dir):
        # network
        self.model = model

        # move the network to GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

        # optimizer
        if config['optimizer']['type'] == 'Adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=config['optimizer']['lr'],
                                              weight_decay=config['optimizer']['weight_decay'])
        elif config['optimizer']['type'] == 'SGD':
            self.optimizer = torch.optim.SGD(model.parameters(),lr=config['optimizer']['lr'],
                                             weight_decay=config['optimizer']['weight_decay'],
                                             momentum=config['optimizer']['momentum'])

        self.start_epoch = 0
        self.min_val_loss = np.Inf
        self.result_path = session_path
        self.checkpoint_path = '{}/{}'.format(checkpoint_dir, 'chkp.tar')

        # load network parameters and optimizer if resuming from previous session
        if config['previous_session'] != "":
            resume_path = "{}/{}/{}".format(result_path, config['previous_session'], 'chkp.tar')
            self.resume_checkpoint(resume_path)

        # data loaders
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # store avg loss and errors fro epochs
        self.epoch_loss_train = {}
        self.epoch_error_train = None
        self.epoch_loss_valid = {}
        self.epoch_error_valid = None

        # config dictionary
        self.config = config
        self.max_epochs = self.config['trainer']['max_epochs']
        # self.lr_decay_ee = self.config['optimizer']['lr_decay_ee']
        self.lr_decays = {}  # {i: 0.1 ** (1 / self.lr_decay_ee) for i in range(1, self.max_epochs)}

    def train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        """
        total_loss = {}
        total_sq_error = torch.zeros(6)

        self.model.train()

        t = [time.time()]
        last_display = time.time()
        start = time.time()

        count = 0
        with torch.enable_grad():
            with torch.autograd.set_detect_anomaly(self.config['detect_anomaly']):

                for i_batch, batch_sample in enumerate(self.train_loader):
                    self.optimizer.zero_grad()

                    loss = self.model(batch_sample, epoch, i_batch)
                    t += [time.time()]

                    loss['LOSS'].backward()

                    self.optimizer.step()

                    # record prediction error for each DOF
                    if epoch >= self.config['loss']['start_svd_epoch']:
                        total_sq_error += torch.sum(self.model.get_error()**2, dim=0)

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
        print('\n Training epoch: {}'.format(epoch))

        for key in total_loss:
            avg_loss = total_loss[key] / count
            self.epoch_loss_train[key].append(avg_loss)
            print('{}: {:.6f}'.format(key, avg_loss))

        num_samples = count * self.config['train_loader']['batch_size']
        rms_error = torch.sqrt(total_error / num_samples)
        if self.epoch_error_train is None:
            self.epoch_error_train = rms_error
        else:
            self.epoch_error_train = np.concatenate((self.epoch_errors, rms_error), axis=0)

        print('Avg error: {}'.format(self.epoch_error[-1, :]))
        print('Time: {}'.format(time.time() - start))
        print('\n')

    def valid_epoch(self, epoch):
        """
        Validation logic for an epoch
        :param epoch: Integer, current training epoch.
        """
        total_loss = {}
        total_sq_error = torch.zeros(6)

        self.model.eval()

        t = [time.time()]
        last_display = time.time()
        start = time.time()

        count = 0
        with torch.no_grad():

                for i_batch, batch_sample in enumerate(self.valid_loader):

                    loss = self.model(batch_sample, epoch, i_batch)
                    t += [time.time()]

                    # record prediction error for each DOF
                    if epoch >= self.config['loss']['start_svd_epoch']:
                        total_sq_error += torch.sum(self.model.get_error()**2, dim=0)

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
        print('\n Validation epoch: {}'.format(epoch))

        for key in total_loss:
            avg_loss = total_loss[key] / count
            self.epoch_loss_valid[key].append(avg_loss)
            print('{}: {:.6f}'.format(key, avg_loss))

        num_samples = count * self.config['train_loader']['batch_size']
        rms_error = torch.sqrt(total_error / num_samples)
        print('Avg error: {}'.format(avg_error))
        if self.epoch_error_valid is None:
            self.epoch_error_valid = rms_error
        else:
            self.epoch_error_valid = np.concatenate((self.epoch_errors, rms_error), axis=0)

        print('Time: {}'.format(time.time() - start))
        print('\n')

        return self.epoch_loss_valid['LOSS'][-1]

    def train(self):
        """
        Full training loop
        """
        # validation and early stopping
        early_stopping = None
        if self.config['trainer']['validate']['on']:
            early_stopping = EarlyStopping(patience=self.config['trainer']['validate']['patience'],
                                           val_loss_min=self.min_val_loss)

        for epoch in range(self.start_epoch, self.config['trainer']['max_epochs']):

            self.train_epoch(epoch)

            if epoch in self.lr_decays:
                for param_group in self.optimizer.param_groups:
                    # param_group['lr'] *= config.lr_decays[self.epoch]
                    param_group['lr'] = self.config['optimizer']['lr'] * self.lr_decays[epoch]
                print("Current epoch learning rate: {}".format(self.config['optimizer']['lr'] * self.lr_decays[epoch]))

            if self.config['trainer']['validate']['on']:
                # check for validation set and early stopping
                val_loss = self.valid_epoch(epoch)
                stop_flag, loss_decrease_flag, self.min_val_loss = early_stopping.check_stop(val_loss, self.model,
                                                                        self.optimizer, self.checkpoint_path, epoch)

                if stop_flag:
                    break
            else:
                # save out every epoch if no validation
                val_loss = None
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': val_loss,
                            }, self.checkpoint_path)

            plot_epoch_losses(self.epoch_loss_train, self.epoch_loss_valid, self.result_path)
            if epoch >= self.config['loss']['start_svd_epoch']:
                plot_epoch_errors(self.epoch_error_train, self.epoch_error_valid, self.result_path)

    def resume_checkpoint(self, checkpoint_path):
        """
        Resume from saved checkpoint
        :param checkpoint_path: Modol filename and path to be resumed
        """

        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.min_val_loss = checkpoint['loss']

        print("Resume training from epoch {}".format(self.start_epoch))

