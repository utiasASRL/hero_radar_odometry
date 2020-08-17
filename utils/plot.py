import sys
import os

import torch
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def plot_epoch_losses(epoch_losses_train, epoch_losses_valid, results_dir):

    for k in epoch_losses_train.keys():
        plt.figure()
        p1 = plt.plot(epoch_losses_train[k])
        p2 = plt.plot(epoch_losses_valid[k])

        plt.legend((p1[0], p2[0]), ('training', 'validation'))
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.title('Loss for each epoch, {}'.format(k))

        plt.savefig(results_dir + 'loss_epoch_' + k + '.png', format='png')
        plt.close()

        plt.figure()
        p1 = plt.plot(np.log(epoch_losses_train[k]))
        p2 = plt.plot(np.log(epoch_losses_valid[k]))

        plt.legend((p1[0], p2[0]), ('training', 'validation'))
        plt.ylabel('Log of loss')
        plt.xlabel('Epoch')
        plt.title('Log of loss for each epoch, {}'.format(k))

        plt.savefig(results_dir + 'loss_log_epoch_' + k + '.png', format='png')
        plt.close()

def plot_epoch_errors(epoch_error_train, epoch_error_valid, results_dir):
    dof = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']

    for i in range(len(dof)):
        plt.figure()
        p1 = plt.plot(epoch_error_train[:, i])
        p2 = plt.plot(epoch_error_valid[:, i])

        plt.legend((p1[0], p2[0]), ('training', 'validation'))
        plt.ylabel('RMSE')
        plt.xlabel('Epoch')
        plt.title('Error for each epoch - ' + dof[i])

        plt.savefig(results_dir + 'error_epoch_' + dof[i] + '.png', format='png')
        plt.close()