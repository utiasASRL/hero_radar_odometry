import os
from time import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.utils import computeMedianError, computeKittiMetrics, get_inverse_tf
from utils.losses import supervised_loss, unsupervised_loss
from utils.utils import get_T_ba
from utils.vis import draw_batch, plot_sequences, draw_batch_steam

class MonitorBase(object):
    """This base class is used for monitoring the training process and executing validation / visualization."""
    def __init__(self, model, valid_loader, config):
        self.model = model
        self.log_dir = config['log_dir']
        self.valid_loader = valid_loader
        self.seq_lens = valid_loader.dataset.seq_lens
        self.sequences = valid_loader.dataset.sequences
        self.config = config
        self.gpuid = config['gpuid']
        self.counter = 0
        self.dt = 0
        self.current_time = 0
        self.vis_batches = self.get_vis_batches()
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        print('monitor running and saving to {}'.format(self.log_dir))

    def get_vis_batches(self):
        """Returns a list of batch indices which we will use for visualization during validation."""
        return np.linspace(0, len(self.valid_loader.dataset) - 1, self.config['vis_num']).astype(np.int32)

    def step(self, loss, dict_loss):
        """At each step of the monitor, we can print, log, validate, or save model information."""
        self.counter += 1
        self.dt = time() - self.current_time
        self.current_time = time()
        valid_metric = None

        if self.counter % self.config['print_rate'] == 0:
            print('Batch: {}\t\t| Loss: {}\t| Step time: {}'.format(self.counter, loss.detach().cpu().item(), self.dt))

        if self.counter % self.config['log_rate'] == 0:
            self.writer.add_scalar('train/loss', loss.detach().cpu().item(), self.counter)
            for loss_name in dict_loss:
                self.writer.add_scalar('train/' + loss_name, dict_loss[loss_name].detach().cpu().item(), self.counter)
            self.writer.add_scalar('train/step_time', self.dt, self.counter)

        if self.counter % self.config['val_rate'] == 0:
            with torch.no_grad():
                self.model.eval()
                valid_metric = self.validation()
                self.model.train()

        return valid_metric

    def vis(self, batchi, batch, out):
        """Visualizes the output from a single batch."""
        raise NotImplementedError('Subclasses must override vis()!')

    def validation(self):
        """This function will compute loss, median errors, KITTI metrics, and draw visualizations."""
        raise NotImplementedError('Subclasses must override validation()!')

class SVDMonitor(MonitorBase):

    def vis(self, batchi, batch, out):
        """Visualizes the output from a single batch."""
        batch_img = draw_batch(batch, out, self.config)
        self.writer.add_image('val/batch_img/{}'.format(batchi), batch_img)

    def validation(self):
        """This function will compute loss, median errors, KITTI metrics, and draw visualizations."""
        time_used = []
        valid_loss = 0
        aux_losses = {}
        aux_init = False
        T_gt = []
        T_pred = []
        for batchi, batch in enumerate(self.valid_loader):
            ts = time()
            if (batchi + 1) % self.config['print_rate'] == 0:
                print('Eval Batch {}: {:.2}s'.format(batchi, np.mean(time_used[-self.config['print_rate']:])))
            with torch.no_grad():
                out = self.model(batch)
            if batchi in self.vis_batches:
                self.vis(batchi, batch, out)
            loss, dict_loss = supervised_loss(out['R'], out['t'], batch, self.config)
            valid_loss += loss.detach().cpu().item()
            if not aux_init:
                for loss_name in dict_loss:
                    aux_losses[loss_name] = dict_loss[loss_name].detach().cpu().item()
                aux_init = True
            else:
                for loss_name in dict_loss:
                    aux_losses[loss_name] += dict_loss[loss_name].detach().cpu().item()
            time_used.append(time() - ts)
            T_gt.append(batch['T_21'][0].numpy().squeeze())
            R_pred_ = out['R'][0].detach().cpu().numpy().squeeze()
            t_pred_ = out['t'][0].detach().cpu().numpy().squeeze()
            T_pred.append(get_transform2(R_pred_, t_pred_))

        results = computeMedianError(T_gt, T_pred)
        t_err, r_err, _ = computeKittiMetrics(T_gt, T_pred, self.seq_lens)

        self.writer.add_scalar('val/loss', valid_loss, self.counter)
        for loss_name in aux_losses:
            self.writer.add_scalar('val/' + loss_name, aux_losses[loss_name], self.counter)
        self.writer.add_scalar('val/avg_time_per_batch', sum(time_used)/len(time_used), self.counter)
        self.writer.add_scalar('val/t_err_med', results[0], self.counter)
        self.writer.add_scalar('val/t_err_std', results[1], self.counter)
        self.writer.add_scalar('val/R_err_med', results[2], self.counter)
        self.writer.add_scalar('val/R_err_std', results[3], self.counter)
        self.writer.add_scalar('val/KITTI/t_err', t_err, self.counter)
        self.writer.add_scalar('val/KITTI/r_err', r_err, self.counter)

        imgs = plot_sequences(T_gt, T_pred, self.seq_lens)
        for i, img in enumerate(imgs):
            self.writer.add_image('val/' + self.sequences[i], img)
        return t_err

class SteamMonitor(MonitorBase):

    def step(self, total_loss, dict_loss):
        """At each step of the monitor, we can print, log, validate, or save model information."""
        self.counter += 1
        self.dt = time() - self.current_time
        self.current_time = time()
        losses = dict_loss.items()
        valid_metric = None

        if self.counter % self.config['print_rate'] == 0:
            print('Batch: {}\t\t| Loss: {}\t| Step time: {}'.format(self.counter, total_loss.detach().cpu().item(), self.dt))

        if self.counter % self.config['log_rate'] == 0:
            self.writer.add_scalar('train/loss', total_loss.detach().cpu().item(), self.counter)
            for loss_item in losses:
                self.writer.add_scalar('train/' + loss_item[0], loss_item[1].detach().cpu().item(), self.counter)
            self.writer.add_scalar('train/step_time', self.dt, self.counter)

        if self.counter % self.config['val_rate'] == 0:
            with torch.no_grad():
                self.model.eval()
                self.model.solver.sliding_flag = True
                valid_metric = self.validation()
                self.model.solver.sliding_flag = False
                self.model.train()

        return valid_metric

    def vis(self, batchi, batch, out):
        """Visualizes the output from a single batch."""
        score_img, match_img, error_img = draw_batch_steam(batch, out, self.config)
        self.writer.add_image('val/score_img/{}'.format(batchi), score_img, global_step=self.counter)
        self.writer.add_image('val/match_img/{}'.format(batchi), match_img, global_step=self.counter)
        self.writer.add_image('val/error_img/{}'.format(batchi), error_img, global_step=self.counter)

    def validation(self):
        """This function will compute loss, median errors, KITTI metrics, and draw visualizations."""
        time_used = []
        valid_loss = 0
        aux_losses = {}
        aux_init = False
        T_gt = []
        T_pred = []

        for batchi, batch in enumerate(self.valid_loader):
            ts = time()
            if (batchi + 1) % self.config['print_rate'] == 0:
                print('Eval Batch {}: {:.2}s'.format(batchi, np.mean(time_used[-self.config['print_rate']:])))
            try:
                out = self.model(batch)
            except RuntimeError as e:
                print(e)
                print('WARNING: exception encountered... skipping this batch.')
                continue
            if batchi in self.vis_batches:
                self.vis(batchi, batch, out)
            loss, dict_loss = unsupervised_loss(out, batch, self.config, self.model.solver)
            if loss != 0:
                valid_loss += loss.detach().cpu().item()
                if not aux_init:
                    for loss_name in dict_loss:
                        aux_losses[loss_name] = dict_loss[loss_name].detach().cpu().item()
                    aux_init = True
                else:
                    for loss_name in dict_loss:
                        aux_losses[loss_name] += dict_loss[loss_name].detach().cpu().item()
            time_used.append(time() - ts)
            if batchi == len(self.valid_loader) - 1:
                # append entire window
                for w in range(batch['T_21'].size(0)-1):
                    T_gt.append(batch['T_21'][w].numpy().squeeze())
                    T_pred.append(get_T_ba(out, a=w, b=w+1))
            else:
                # append only the front of window
                T_gt.append(batch['T_21'][0].numpy().squeeze())
                T_pred.append(get_T_ba(out, a=0, b=1))

        results = computeMedianError(T_gt, T_pred)
        t_err, r_err, _ = computeKittiMetrics(T_gt, T_pred, self.seq_lens)

        self.writer.add_scalar('val/loss', valid_loss, self.counter)
        for loss_name in aux_losses:
            self.writer.add_scalar('val/' + loss_name, aux_losses[loss_name], self.counter)
        self.writer.add_scalar('val/avg_time_per_batch', sum(time_used)/len(time_used), self.counter)
        self.writer.add_scalar('val/t_err_med', results[0], self.counter)
        self.writer.add_scalar('val/t_err_std', results[1], self.counter)
        self.writer.add_scalar('val/R_err_med', results[2], self.counter)
        self.writer.add_scalar('val/R_err_std', results[3], self.counter)
        self.writer.add_scalar('val/t_err_mean', results[4], self.counter)
        self.writer.add_scalar('val/R_err_mean', results[5], self.counter)
        self.writer.add_scalar('val/KITTI/t_err', t_err, self.counter)
        self.writer.add_scalar('val/KITTI/r_err', r_err, self.counter)

        imgs = plot_sequences(T_gt, T_pred, self.seq_lens)
        for i, img in enumerate(imgs):
            self.writer.add_image('val/' + self.sequences[i], img, self.counter)
        return t_err
