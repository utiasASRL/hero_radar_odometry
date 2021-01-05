from utils.monitor_base import MonitorBase
from time import time
import numpy as np
from utils.utils import computeMedianError, computeKittiMetrics
from utils.vis import plot_sequences
from utils.vis import draw_batch_steam

class SteamMonitor(MonitorBase):
    def __init__(self, model, valid_loader, config):
        super().__init__(model, valid_loader, config)

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
            loss, dict_loss = self.model.loss(out['src'], out['tgt'], out['match_weights'])
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

    def vis(self, batchi, batch, out):
        """Visualizes the output from a single batch."""
        batch_img = draw_batch_steam(batch, out, self.config)
        self.writer.add_image('val/batch_img/{}'.format(batchi), batch_img)