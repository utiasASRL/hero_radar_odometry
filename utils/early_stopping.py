import numpy as np
import torch

class EarlyStopping:
    """Based on https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
       Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=True, val_loss_min=np.Inf):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.val_loss_min = val_loss_min
        self.epoch_min = 0

    def check_stop(self, val_loss, model, optimizer, model_file_name, epoch):

        if val_loss > self.val_loss_min:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            print(f'Current val loss: {val_loss:.10f}, min val loss: {self.val_loss_min:.10f}\n')
            if self.counter >= self.patience:
                print('Early stopping: model saved at epoch {}'.format(self.epoch_min))
                return True, self.counter == 0, self.val_loss_min
        else:
            self.save_checkpoint(val_loss, model, optimizer, model_file_name, epoch)
            self.counter = 0

        return False, self.counter == 0, self.val_loss_min

    def save_checkpoint(self, val_loss, model, optimizer, model_file_name, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model at epoch {epoch}.\n')

        self.val_loss_min = val_loss
        self.epoch_min = epoch

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    }, model_file_name)