import os
from time import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.utils import supervised_loss, pointmatch_loss, computeMedianError, computeKittiMetrics, get_inverse_tf
from utils.vis import draw_batch, plot_sequences, draw_batch_steam, draw_batch_steam_eval, draw_mah_histogram
from datasets.transforms import augmentBatch

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
        valid_loss = None

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
                valid_loss = self.validation()
                self.model.train()

        if self.counter % self.config['save_rate'] == 0:
            with torch.no_grad():
                self.model.eval()
                mname = os.path.join(self.log_dir, '{}.pt'.format(self.counter))
                print('saving model', mname)
                torch.save(self.model.state_dict(), mname)
                self.model.train()
        return valid_loss

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
                loss, dict_loss = pointmatch_loss(out, batch, self.config)
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
        return valid_loss

class SteamMonitor(MonitorBase):

    def step(self, batchi, total_loss, dict_loss):
        """At each step of the monitor, we can print, log, validate, or save model information."""
        self.counter += 1
        self.dt = time() - self.current_time
        self.current_time = time()
        losses = dict_loss.items()

        if self.counter % self.config['print_rate'] == 0:
            print('Batch: {}\t\t| Loss: {}\t| Step time: {}'.format(batchi, total_loss.detach().cpu().item(), self.dt))

        if self.counter % self.config['log_rate'] == 0:
            self.writer.add_scalar('train/loss', total_loss.detach().cpu().item(), self.counter)
            for loss_item in losses:
                self.writer.add_scalar('train/' + loss_item[0], loss_item[1].detach().cpu().item(), self.counter)
            self.writer.add_scalar('train/step_time', self.dt, self.counter)

        if self.counter % self.config['val_rate'] == 0:
            with torch.no_grad():
                self.model.eval()
                self.model.solver.sliding_flag = True
                self.validation()
                self.model.solver.sliding_flag = False
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
        score_img, match_img, error_img = draw_batch_steam(batch, out, self.config)
        self.writer.add_image('val/score_img/{}'.format(batchi), score_img, global_step=self.counter)
        self.writer.add_image('val/match_img/{}'.format(batchi), match_img, global_step=self.counter)
        self.writer.add_image('val/error_img/{}'.format(batchi), error_img, global_step=self.counter)

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
            #loss, dict_loss = self.model.loss(out['src'], out['tgt'], out['match_weights'], out['keypoint_ints'])
            loss, dict_loss = self.model.loss(out['src'], out['tgt'], out['match_weights'], out['keypoint_ints'], out['scores'], batch)
            if loss != 0:
                valid_loss += loss.detach().cpu().item()
                valid_point_loss += dict_loss['point_loss'].detach().cpu().item()
                valid_logdet_loss += dict_loss['logdet_loss'].detach().cpu().item()
            time_used.append(time() - ts)
            if batchi == 0:
                # append entire window
                for w in range(batch['T_21'].size(0)-1):
                    T_gt.append(batch['T_21'][w].numpy().squeeze())
                    T_pred = get_T_ba(out, a=w, b=w+1)
                    R_pred.append(T_pred[:3, :3].squeeze())
                    t_pred.append(T_pred[:3, 3].squeeze())
            else:
                # append only the front of window
                T_gt.append(batch['T_21'][-2].numpy().squeeze())
                T_pred = get_T_ba(out, a=-2, b=-1)
                R_pred.append(T_pred[:3, :3].squeeze())
                t_pred.append(T_pred[:3, 3].squeeze())

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
            self.writer.add_image('val/' + self.sequences[i], img, self.counter)
        return valid_loss

def get_T_ba(out, a, b):
    T_b0 = np.eye(4)
    T_b0[:3, :3] = out['R'][0, b].detach().cpu().numpy()
    T_b0[:3, 3:4] = out['t'][0, b].detach().cpu().numpy()
    T_a0 = np.eye(4)
    T_a0[:3, :3] = out['R'][0, a].detach().cpu().numpy()
    T_a0[:3, 3:4] = out['t'][0, a].detach().cpu().numpy()
    return T_b0@get_inverse_tf(T_a0)


class SteamEvalMonitor(object):

    def __init__(self, model, valid_loader, train_config, eval_config):
        self.model = model
        self.log_dir = eval_config['log_dir']
        self.valid_loader = valid_loader
        self.seq_len = valid_loader.dataset.seq_len
        self.sequences = valid_loader.dataset.sequences
        self.train_config = train_config
        self.eval_config = eval_config
        self.gpuid = eval_config['gpuid']
        self.counter = 0
        self.dt = 0
        self.current_time = 0
        self.bin_count_total = []

        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        print('monitor running and saving to {}'.format(self.log_dir))

    def vis(self, batchi, batch, out):
        """Visualizes the output from a single batch."""
        radar_img, match_img = draw_batch_steam_eval(batch, out, self.model)
        self.writer.add_image('val/radar_img', radar_img, global_step=batchi)
        self.writer.add_image('val/match_img', match_img, global_step=batchi)
        # self.writer.add_image('val/points_img', points_img, global_step=batchi)

    def evaluate(self):
        self.model.eval()
        self.model.solver.sliding_flag = True
        with torch.no_grad():
            time_used = []
            T_gt = []
            R_pred = []
            t_pred = []
            for batchi, batch in enumerate(self.valid_loader):
                if self.eval_config['augmentation']['augment']:
                    batch = augmentBatch(batch, self.eval_config)
                ts = time()
                if (batchi + 1) % self.eval_config['print_rate'] == 0:
                    print('Eval Batch {}: {:.2}s'.format(batchi, np.mean(time_used[-self.eval_config['print_rate']:])))
                out = self.model(batch)

                # mah stats
                self.mah_stats(out, batch)
                # mah_hist_img = draw_mah_histogram(self.bin_count_total)
                # self.writer.add_image('val/mah_hist', mah_hist_img, global_step=batchi)

                if batchi % self.eval_config['vis_skip'] == 0:
                    self.vis(batchi, batch, out)
                    mah_hist_img = draw_mah_histogram(self.bin_count_total)
                    self.writer.add_image('val/mah_hist', mah_hist_img, global_step=batchi)

                time_used.append(time() - ts)
                if batchi == 0:
                    # append entire window
                    for w in range(batch['T_21'].size(0)-1):
                        T_gt.append(batch['T_21'][w].numpy().squeeze())
                        T_pred = get_T_ba(out, a=w, b=w+1)
                        R_pred.append(T_pred[:3, :3].squeeze())
                        t_pred.append(T_pred[:3, 3].squeeze())
                else:
                    # append only the front of window
                    T_gt.append(batch['T_21'][-2].numpy().squeeze())
                    T_pred = get_T_ba(out, a=-2, b=-1)
                    R_pred.append(T_pred[:3, :3].squeeze())
                    t_pred.append(T_pred[:3, 3].squeeze())

            t_err, r_err, _ = computeKittiMetrics(T_gt, R_pred, t_pred, self.seq_len)

            self.writer.add_scalar('val/KITTI/t_err', t_err, self.counter)
            self.writer.add_scalar('val/KITTI/r_err', r_err, self.counter)

            imgs = plot_sequences(T_gt, R_pred, t_pred, self.seq_len)
            for i, img in enumerate(imgs):
                self.writer.add_image('val/' + self.sequences[i], img, self.counter)

            # plot mah histogram
            mah_hist_img = draw_mah_histogram(self.bin_count_total)
            self.writer.add_image('val/mah_hist', mah_hist_img, self.counter)

        self.model.solver.sliding_flag = False

    def mah_stats(self, out, batch):
        src_coords = out['src']
        tgt_coords = out['tgt']
        match_weights = out['match_weights']
        keypoint_ints = out['keypoint_ints']

        # setup histogram parameters
        # assume delta is always 1
        max_bin_val = 10    # must be int
        if not self.bin_count_total:
            self.bin_count_total = [np.zeros(max_bin_val+1)]*(self.model.solver.window_size - 1)

        # get first pose T_10 (using groundtruth)
        T_k0 = batch['T_21'][0].to(self.gpuid)

        # loop for each window frame (assume batch_size of 1)
        for w in range(self.model.solver.window_size - 1):
            # filter by zero intensity patches
            ids = torch.nonzero(keypoint_ints[w, 0] > 0, as_tuple=False).squeeze(1)

            # points must be list of N x 3
            zeros_vec = torch.zeros_like(src_coords[w, ids, 0:1])
            points1 = torch.cat((src_coords[w, ids], zeros_vec), dim=1).unsqueeze(-1)    # N x 3 x 1
            points2 = torch.cat((tgt_coords[w, ids], zeros_vec), dim=1).unsqueeze(-1)    # N x 3 x 1
            weights_mat, _ = self.model.solver.convert_to_weight_matrix(match_weights[w, :, ids].T, w)  # N x 3 x 3

            # get R_21 and t_12_in_2
            R_21 = T_k0[:3, :3]
            t_12_in_2 = T_k0[:3, 3:4]

            # compute mah
            error = points2 - (R_21 @ points1 + t_12_in_2)
            mah2_error = error.transpose(1, 2)@weights_mat@error
            mah2_bin_vals = np.floor(np.clip(mah2_error.squeeze().detach().cpu().numpy(), 0, max_bin_val))
            bin_count = np.bincount(mah2_bin_vals.astype(int), minlength=max_bin_val+1)
            self.bin_count_total[w] += bin_count

            # update T_k0
            T_kp1_k = batch['T_21'][w+1].to(self.gpuid)
            T_k0 = T_kp1_k@T_k0

        return
