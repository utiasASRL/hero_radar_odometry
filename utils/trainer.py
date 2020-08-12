import os
import sys
import time
import torch
import numpy as np

from utils.early_stopping import EarlyStopping
from utils.lie_algebra import so3_to_rpy

class Trainer():
    """
    Trainer class
    """
    def __init__(self, model, train_loader, valid_loader, config):
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
        self.result_path = os.path.join('results', config['session_name'])
        self.checkpoint_path = os.path.join(self.result_path, 'checkpoints/chkp.tar')
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        # load network parameters and optimizer if resuming from previous session
        if config['previous_session'] != "":
            resume_path = "{}/{}/{}".format('results', config['previous_session'], 'chkp.tar')
            self.resume_checkpoint(resume_path)

        # data loaders
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # config dictionary
        self.config = config
        self.max_epochs = self.config['trainer']['max_epochs']
        self.lr_decay_ee = self.config['optimizer']['lr_decay_ee']
        self.lr_decays = {i: 0.1 ** (1 / self.lr_decay_ee) for i in range(1, self.max_epochs)}

        # logging
        self.stdout_orig = sys.stdout
        self.log_path = os.path.join(self.result_path, 'train.txt')
        self.stdout_file = open(self.log_path, 'w')

    def train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        """

        self.model.train()

        # TODO: add anomoly detection
        t = [time.time()]
        last_display = time.time()
        with torch.enable_grad():
            for i_batch, batch_sample in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                # TODO: forward prop
                loss = self.model(batch_sample)
                t += [time.time()]

                loss['LOSS'].backward()

                self.optimizer.step()

                # save intermediate outputs
                self.model.save_intermediate_outputs()

                # Console print (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    sys.stdout = self.stdout_orig
                    self.model.print_loss(loss, epoch, i_batch)

                # # File print (every time)
                # sys.stdout = self.stdout_file
                # self.model.print_loss(loss, epoch, i_batch)
                # self.stdout_file.flush()

        return loss

    def valid_epoch(self, epoch):
        self.model.eval()

        # init saving lists
        rotations_ab = [] # store R_21
        translations_ab = [] # store t_21
        rotations_ab_pred = [] # store R_21_pred
        translations_ab_pred = [] # store t_21_pred

        eulers_ab = [] # store euler_12
        eulers_ab_pred = [] # store euler_12_pred

        with torch.no_grad():
            for i_batch, batch_sample in enumerate(self.test_loader):

                # forward prop
                loss = self.model(batch_sample)
                self.textio.cprint("loss:{}".format(loss['LOSS'].item()))
                self.test_loss += loss['LOSS'].item()

                # collect poses
                save_dict = self.model.return_save_dict()
                R_21_pred, t_21_pred = save_dict['R_pred'], save_dict['t_pred']
                R_21, t_21 = save_dict['R_tgt_src'], save_dict['t_tgt_src']
                euler_21 = so3_to_rpy(R_21)
                euler_21_pred = so3_to_rpy(R_21_pred)

                # append to list
                rotations_ab.append(R_21.detach().cpu().numpy())
                translations_ab.append(t_21.detach().cpu().numpy())
                rotations_ab_pred.append(R_21_pred.detach().cpu().numpy())
                translations_ab_pred.append(t_21_pred.detach().cpu().numpy())

                eulers_ab.append(euler_21.detach().cpu().numpy())
                eulers_ab_pred.append(euler_21_pred.detach().cpu().numpy())

            # concat and convert to numpy array
            rotations_ab = np.concatenate(rotations_ab, axis=0)
            translations_ab = np.concatenate(translations_ab, axis=0)
            rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
            translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

            eulers_ab = np.concatenate(eulers_ab, axis=0)
            eulers_ab_pred = np.concatenate(eulers_ab_pred, axis=0)

        test_r_mse_ab = np.mean((eulers_ab_pred - eulers_ab) ** 2)
        test_r_rmse_ab = np.sqrt(test_r_mse_ab)
        test_r_mae_ab = np.mean(np.abs(eulers_ab_pred - eulers_ab))
        test_t_mse_ab = np.mean((translations_ab - translations_ab_pred) ** 2)
        test_t_rmse_ab = np.sqrt(test_t_mse_ab)
        test_t_mae_ab = np.mean(np.abs(translations_ab - translations_ab_pred))

        self.textio.cprint('==FINAL TEST==')
        self.textio.cprint('A--------->B')
        self.textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, '
                           'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                           % (-1, self.test_loss, test_r_mse_ab, test_r_rmse_ab, test_r_mae_ab,
                              test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))


        return loss['LOSS'].item()

    def train(self):
        """
        Full training loop
        """
        # validation and early stopping
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
                stop_flag, loss_decrease_flag, self.min_val_loss = early_stopping.check_stop(
                    val_loss, self.model, self.optimizer, self.checkpoint_path, epoch)

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

        # close log file
        sys.stdout = self.stdout_orig
        self.stdout_file.close()

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

