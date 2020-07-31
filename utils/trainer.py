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
        # TODO: load network parameters if resuming
        # TODO: load optimizer states if resuming

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
        with torch.enable_grad():
            for i_batch, batch_sample in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                # TODO: forward prop

                # TODO: backwards prop

    def train(self):
        """
        Full training loop
        """
        # TODO: validation and early stopping
        # early_stopping = EarlyStopping(patience=self.config['trainer']['patience'],
        #                                best_score=-self.min_val_loss,
        #                                val_loss_min=self.min_val_loss)

        for epoch in range(self.start_epoch, self.config['trainer']['max_epochs']):

            self.train_epoch(epoch)
            # val_loss = self.valid_epoch(epoch, self.valid_data_loader,
            #                             self.valid_stats, 'validation')

