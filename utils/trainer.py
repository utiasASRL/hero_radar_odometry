import os
import time
import torch
import numpy as np

from utils.early_stopping import EarlyStopping

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
        self.checkpoint_path = os.path.join(self.result_path, 'chkp.tar')

        # load network parameters and optimizer if resuming from previous session
        if config['previous_session'] != "":
            resume_path = "{}/{}/{}".format('results', config['previous_session'], 'chkp.tar')
            self.resume_checkpoint(resume_path)

        # data loaders
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # config dictionary
        self.config = config

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

                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    self.model.print_loss(loss)

                # TODO: backwards prop

        return loss

    def valid_epoch(self, epoch):
        self.model.eval()

        # TODO: add anomoly detection
        loss = 0
        with torch.no_grad():
            for i_batch, batch_sample in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                # TODO: forward prop
                loss += 0    # summation!


        return loss

    def train(self):
        """
        Full training loop
        """
        # validation and early stopping
        if self.config['trainer']['validate']['on']:
            train_loss = early_stopping = EarlyStopping(patience=self.config['trainer']['validate']['patience'],
                                                        val_loss_min=self.min_val_loss)

        for epoch in range(self.start_epoch, self.config['trainer']['max_epochs']):

            self.train_epoch(epoch)

            if self.config['validate']['on']:
                # check for validation set and early stopping
                val_loss = self.valid_epoch(epoch)
                stop_flag, loss_decrease_flag, self.min_val_loss = early_stopping.check_stop(
                    val_loss, self.model, self.optimizer, self.checkpoint_path, epoch)

                if stop_flag:
                    break
            else:
                # save out every epoch if no validation
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': val_loss,
                            }, self.checkpoint_path)

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

