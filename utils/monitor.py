import os
import torch
import numpy as np
from tensorboardX import SummaryWriter
from time import time

from networks.svd_pose_model import SVDPoseModel
from utils.utils import supervised_loss, computeMedianError, computeKittiMetrics
from utils.vis import draw_batch, plot_sequences

class Monitor(object):
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
        return np.linspace(0, len(self.valid_loader.dataset) - 1, self.config['vis_num']).astype(np.int32)

    def step(self, batchi, loss, R_loss, t_loss, batch, out):
        self.counter += 1
        self.dt = time() - self.current_time
        self.current_time = time()

        if self.counter % self.config['print_rate'] == 0:
            print('Batch: {}\t\t| Loss: {}\t| Step time: {}'.format(batchi, loss.detach().cpu().item(), self.dt))

        if self.counter % self.config['log_rate'] == 0:
            self.writer.add_scalar('train/loss', loss.detach().cpu().item(), self.counter)
            self.writer.add_scalar('train/R_loss', R_loss.detach().cpu().item(), self.counter)
            self.writer.add_scalar('train/t_loss', t_loss.detach().cpu().item(), self.counter)
            self.writer.add_scalar('train/step_time', self.dt, self.counter)

        if self.counter % self.config['val_rate'] == 0:
            with torch.no_grad():
                self.model.eval()
                self.validation()
                self.model.train()

        if self.counter % self.config['save_rate'] == 0:
            with torch.no_grad():
                self.model.eval()
                mname = os.path.join(self.log_dir, "{}.pt".format(self.counter))
                print('saving model', mname)
                torch.save(self.model.state_dict(), mname)
                self.model.train()

    def vis(self, batchi, batch, out):
        batch_img = draw_batch(batch, out, self.config)
        self.writer.add_image('val/batch_img/{}'.format(batchi), batch_img)

    def validation(self):
        time_used = []self.writer.add_image('val/' )
        valid_loss = 0
        valid_R_loss = 0
        valid_t_loss = 0
        T_gt = []
        R_pred = []
        t_pred = []
        for batchi, batch in enumerate(self.valid_loader):
            ts = time()
            if (batchi + 1) % self.config['print_rate'] == 0:
                print("Eval Batch {}: {:.2}s".format(batchi, np.mean(time_used[-self.config['print_rate']:])))
            out = self.model(batch)
            if batchi in self.vis_batches:
                self.vis(batchi, batch, out)
            loss, R_loss, t_loss = supervised_loss(out['R'], out['t'], batch, self.config)
            valid_loss += loss.detach().cpu().item()
            valid_R_loss += R_loss.detach().cpu().item()
            valid_t_loss += t_loss.detach().cpu().item()
            time_used.append(time() - ts)
            T_gt.append(batch['T_21'])
            R_pred.append(out['R'].detach().cpu().numpy())
            t_pred.append(out['t'].detach().cpu().numpy())

        results = computeMedianError(T_gt, R_pred, t_pred)
        t_err, r_err = computeKittiMetrics(T_gt, R_pred, t_pred, self.seq_len)

        self.writer.add_scalar('val/loss', valid_loss, self.counter)
        self.writer.add_scalar('val/R_loss', valid_R_loss, self.counter)
        self.writer.add_scalar('val/t_loss', valid_t_loss, self.counter)
        self.writer.add_scalar('val/avg_time_per_batch', sum(time_used)/len(time_used), self.counter)
        self.writer.add_scalar('val/t_err_med', results[0])
        self.writer.add_scalar('val/t_err_std', results[1])
        self.writer.add_scalar('val/R_err_med', results[2])
        self.writer.add_scalar('val/R_err_std', results[3])

        imgs = plot_sequences(T_gt, R_pred, t_pred, self.seq_len)
        for i in range(len(imgs)):
            self.writer.add_image('val/' + self.sequences[i], imgs[i])
