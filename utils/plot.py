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

        plt.savefig(results_dir + '/loss_epoch_' + k + '.png', format='png')
        plt.close()

        plt.figure()
        p1 = plt.plot(np.log(epoch_losses_train[k]))
        p2 = plt.plot(np.log(epoch_losses_valid[k]))

        plt.legend((p1[0], p2[0]), ('training', 'validation'))
        plt.ylabel('Log of loss')
        plt.xlabel('Epoch')
        plt.title('Log of loss for each epoch, {}'.format(k))

        plt.savefig(results_dir + '/loss_log_epoch_' + k + '.png', format='png')
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

        plt.savefig(results_dir + '/error_epoch_' + dof[i] + '.png', format='png')
        plt.close()

def plot_error(gt, est, titles, setting, save_name):
    '''
    Plot errors between ground truth and estimates.
    @param gt: ground truth containing Nx3 array
    @param est: estimation containing Nx3 array
    @param titles: titles for plotting. Example: ['psi', 'theta', 'phi']
    @param setting: relative or absolute. Option: 'rel', 'abs'
    @param save_name: absolute file path to save
    @return: None
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    # ax1.plot(psi, label='ground truth')
    # ax1.plot(psi_pred, label='prediction')
    ax1.plot(est[:,0] - gt[:,0])
    # ax1.legend()
    # ax1.set_ylim(-0.1, 0.1)
    ax1.set_title('{} error along {}'.format(setting, titles[0]))
    ax2 = fig.add_subplot(312)
    # ax2.plot(theta, label='ground truth')
    # ax2.plot(theta_pred, label='prediction')
    ax2.plot(est[:,1] - gt[:,1])
    # ax2.set_ylim(-0.1, 0.1)
    ax2.set_title('{} error along {}'.format(setting, titles[1]))
    ax3 = fig.add_subplot(313)
    # ax3.plot(phi, label='ground truth')
    # ax3.plot(phi_pred, label='prediction')
    ax3.plot(est[:,2] - gt[:,2])
    # ax3.set_ylim(-0.1, 0.1)
    ax3.set_title('{} error along {}'.format(setting, titles[2]))
    plt.savefig(save_name)

def plot_versus(gt, est, titles, setting, save_name):
    '''
    Plot ground truth and estimates.
    @param gt: ground truth containing Nx3 array
    @param est: estimation containing Nx3 array
    @param titles: titles for plotting. Example: ['psi', 'theta', 'phi']
    @param setting: relative or absolute. Option: 'rel', 'abs'
    @param save_name: absolute file path to save
    @return: None
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax1.plot(gt[:,0], label='ground truth')
    ax1.plot(est[:,0], label='prediction')
    # ax1.plot(est[:,0] - gt[:,0])
    ax1.legend()
    # ax1.set_ylim(-0.1, 0.1)
    ax1.set_title('{} along {}'.format(setting, titles[0]))
    ax2 = fig.add_subplot(312)
    ax2.plot(gt[:,1], label='ground truth')
    ax2.plot(est[:,1], label='prediction')
    # ax2.plot(est[:,1] - gt[:,1])
    ax2.legend()
    # ax2.set_ylim(-0.1, 0.1)
    ax2.set_title('{} along {}'.format(setting, titles[1]))
    ax3 = fig.add_subplot(313)
    ax3.plot(gt[:,2], label='ground truth')
    ax3.plot(est[:,2], label='prediction')
    # ax3.plot(est[:,2] - gt[:,2])
    ax3.legend()
    # ax3.set_ylim(-0.1, 0.1)
    ax3.set_title('{} along {}'.format(setting, titles[2]))
    plt.savefig(save_name)

def plot_route(gt, out, c_gt='g', c_out='r'):
    x_idx = 2
    y_idx = 0
    z_idx = 1

    x = [v for v in gt[:, x_idx]]
    y = [v for v in gt[:, y_idx]]  # TODO dataset specific
    plt.plot(y, x, color=c_gt, label='Ground Truth')
    # plt.scatter(x, y, color='b')

    x = [v for v in out[:, x_idx]]
    y = [-v for v in out[:, y_idx]]
    plt.plot(y, x, color=c_out, label='Estimation')
    # plt.scatter(x, y, color='b')
    plt.gca().set_aspect('equal', adjustable='datalim')