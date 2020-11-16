import argparse
import json

from datasets.oxford import *
from networks.svd_pose_model import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/radar.json', type=str, help='config file path')
    parser.add_argument('--pretrain', default=None, type=str, help='pretrain checkpoint path')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    _, _, test_loader = get_dataloaders(config)

    model = SVDPoseModel(config)
    model.load_state_dict(torch.load(args.pretrain, map_location=torch.device(config['gpuid'])), strict=False)
    model.to(config['gpuid'])
    model.eval()

    time_used = []
    valid_loss = 0
    valid_R_loss = 0
    valid_t_loss = 0
    for batchi, batch in enumerate(test_loader):
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

    print('valid_loss: {}'.format(valid_loss))
    print('valid_R_loss: {}'.format(valid_R_loss))
    print('valid_t_loss: {}'.format(valid_t_loss))
    print('time_used: {}'.format(sum(time_used) / len(time_used)))
