import os
from time import time
import numpy as np
import torch
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from utils.utils import supervised_loss, pointmatch_loss, computeMedianError, computeKittiMetrics
from utils.vis import draw_batch, plot_sequences, draw_batch_steam

class MonitorBase(object):
    """This base class is used for monitoring the training process and executing validation / visualization."""
    def __init__(self, model, valid_loader, config):
        self.model = model
        self.log_dir = config['log_dir']
        self.valid_loader = valid_loader
        self.seq_len = valid_loader.dataset.seq_len
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

    def step(self, batchi, loss, dict_loss):
        """At each step of the monitor, we can print, log, validate, or save model information."""
        self.counter += 1
        self.dt = time() - self.current_time
        self.current_time = time()

        if self.counter % self.config['print_rate'] == 0:
            print('Batch: {}\t\t| Loss: {}\t| Step time: {}'.format(batchi, loss.detach().cpu().item(), self.dt))

        if self.counter % self.config['log_rate'] == 0:
            self.writer.add_scalar('train/loss', loss.detach().cpu().item(), self.counter)
            for loss_name in dict_loss:
                self.writer.add_scalar('train/' + loss_name, dict_loss[loss_name].detach().cpu().item(), self.counter)
            self.writer.add_scalar('train/step_time', self.dt, self.counter)

        if self.counter % self.config['val_rate'] == 0:
            with torch.no_grad():
                self.model.eval()
                self.validation()
                self.model.train()

        if self.counter % self.config['save_rate'] == 0:
            with torch.no_grad():
                self.model.eval()
                mname = os.path.join(self.log_dir, '{}.pt'.format(self.counter))
                print('saving model', mname)
                torch.save(self.model.state_dict(), mname)
                self.model.train()

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
        R_pred = []
        t_pred = []
        for batchi, batch in enumerate(self.valid_loader):
            ts = time()
            if (batchi + 1) % self.config['print_rate'] == 0:
                print('Eval Batch {}: {:.2}s'.format(batchi, np.mean(time_used[-self.config['print_rate']:])))
            out = self.model(batch)
            if batchi in self.vis_batches:
                self.vis(batchi, batch, out)
            if self.config['loss'] == 'supervised_loss':
                loss, dict_loss = supervised_loss(out['R'], out['t'], batch, self.config)
            elif self.config['loss'] == 'pointmatch_loss':
                loss, dict_loss = pointmatch_loss(out['R'], out['t'], out['tgt'], out['src'], out['match_weights'])
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
            R_pred.append(out['R'][0].detach().cpu().numpy().squeeze())
            t_pred.append(out['t'][0].detach().cpu().numpy().squeeze())

        results = computeMedianError(T_gt, R_pred, t_pred)
        t_err, r_err, _ = computeKittiMetrics(T_gt, R_pred, t_pred, self.seq_len)

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

        imgs = plot_sequences(T_gt, R_pred, t_pred, self.seq_len)
        for i, img in enumerate(imgs):
            self.writer.add_image('val/' + self.sequences[i], img)

class SteamMonitor(MonitorBase):

    def vis(self, batchi, batch, out):
        """Visualizes the output from a single batch."""
        batch_img = draw_batch_steam(batch, out, self.config)
        self.writer.add_image('val/batch_img/{}'.format(batchi), batch_img)

    def validation(self):
        """This function will compute loss, median errors, KITTI metrics, and draw visualizations."""
        time_used = []
        valid_loss = 0
        valid_point_loss = 0
        valid_logdet_loss = 0
        T_gt = []
        R_pred = []
        t_pred = []
        for batchi, batch in enumerate(self.valid_loader):
            ts = time()
            if (batchi + 1) % self.config['print_rate'] == 0:
                print('Eval Batch {}: {:.2}s'.format(batchi, np.mean(time_used[-self.config['print_rate']:])))
            out = self.model(batch)
            if batchi in self.vis_batches:
                self.vis(batchi, batch, out)
            loss, dict_loss = self.model.loss(out['src'], out['tgt'], out['match_weights'], out['keypoint_ints'])
            if loss != 0:
                valid_loss += loss.detach().cpu().item()
                valid_point_loss += dict_loss['point_loss'].detach().cpu().item()
                valid_logdet_loss += dict_loss['logdet_loss'].detach().cpu().item()
            time_used.append(time() - ts)
            T_gt.append(batch['T_21'][0].numpy().squeeze())
            R_pred.append(out['R'][0].detach().cpu().numpy().squeeze())
            t_pred.append(out['t'][0].detach().cpu().numpy().squeeze())

        results = computeMedianError(T_gt, R_pred, t_pred)
        t_err, r_err, _ = computeKittiMetrics(T_gt, R_pred, t_pred, self.seq_len)

        self.writer.add_scalar('val/loss', valid_loss, self.counter)
        self.writer.add_scalar('val/point_loss', valid_point_loss, self.counter)
        self.writer.add_scalar('val/logdet_loss', valid_logdet_loss, self.counter)
        self.writer.add_scalar('val/avg_time_per_batch', sum(time_used)/len(time_used), self.counter)
        self.writer.add_scalar('val/t_err_med', results[0], self.counter)
        self.writer.add_scalar('val/t_err_std', results[1], self.counter)
        self.writer.add_scalar('val/R_err_med', results[2], self.counter)
        self.writer.add_scalar('val/R_err_std', results[3], self.counter)
        self.writer.add_scalar('val/KITTI/t_err', t_err, self.counter)
        self.writer.add_scalar('val/KITTI/r_err', r_err, self.counter)

        imgs = plot_sequences(T_gt, R_pred, t_pred, self.seq_len)
        for i, img in enumerate(imgs):
            self.writer.add_image('val/' + self.sequences[i], img)
