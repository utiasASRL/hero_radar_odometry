import os
import sys
import time
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

from utils.lie_algebra import so3_to_rpy
from utils.plot import plot_route, plot_versus, plot_error

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()

class Tester():
    """
    MatchTester class
    """
    def __init__(self, model, test_loader, config):
        # network
        self.model = model

        # move the network to GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

        # load network parameters and optimizer if resuming from previous session
        assert config['previous_session'] != "", "No previous session checkpoint provided!"

        checkpoint_path = "{}/{}/{}/{}".format(config['home_dir'], 'results', config['previous_session'], 'checkpoints/chkp.tar')
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print("Model and training state restored.")

        # data loaders
        self.test_loader = test_loader

        # config dictionary
        self.config = config
        self.window_size = config['test_loader']['window_size']

        # logging
        self.result_path = os.path.join(config['home_dir'], 'results', config['session_name'])
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.plot_path = '{}/{}'.format(self.result_path, 'visualization')
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)
        self.log_path = os.path.join(self.result_path, 'eval.log')
        self.textio = IOStream(self.log_path)
        self.test_loss = 0

    def test_epoch(self):
        self.model.eval()

        # init saving lists
        rotations_ba = [] # store R_ba
        translations_ba = [] # store t_ba
        rotations_ba_pred = [] # store R_ba_pred
        translations_ba_pred = [] # store t_ba_pred
        transforms_ia = []
        transforms_ib = []

        eulers_ba = [] # store euler_ba
        eulers_ba_pred = [] # store euler_ba_pred

        with torch.no_grad():
            for i_batch, batch_sample in enumerate(self.test_loader):

                # if i_batch > 400:
                #     break

                # forward prop
                try:
                    loss = self.model(batch_sample, 0)
                except Exception as e:
                    self.model.print_loss(loss, 0, i_batch)
                    self.model.print_inliers(0, i_batch)
                    print(e)

                if i_batch % 100 == 0:
                    self.model.print_inliers(0, i_batch)

                self.textio.cprint("loss:{}".format(loss['LOSS'].item()))
                self.test_loss += loss['LOSS'].item()

                # collect poses
                save_dict = self.model.return_save_dict()
                R_ba_pred, t_ba_pred = save_dict['R_tgt_src_pred'], save_dict['t_tgt_src_pred']
                R_ba, t_ba = save_dict['R_tgt_src'], save_dict['t_tgt_src']
                T_ia, T_ib = save_dict['T_i_src'], save_dict['T_i_tgt']
                euler_ba = so3_to_rpy(R_ba)
                euler_ba_pred = so3_to_rpy(R_ba_pred)

                # append to list
                rotations_ba.append(R_ba.detach().cpu().numpy())
                translations_ba.append(t_ba.detach().cpu().numpy())
                rotations_ba_pred.append(R_ba_pred.detach().cpu().numpy())
                translations_ba_pred.append(t_ba_pred.detach().cpu().numpy())
                transforms_ia.append(T_ia.detach().cpu().numpy())
                transforms_ib.append(T_ib.detach().cpu().numpy())

                eulers_ba.append(euler_ba.detach().cpu().numpy())
                eulers_ba_pred.append(euler_ba_pred.detach().cpu().numpy())

        # TODO this is a temporary fix for averaging out the test loss
        self.test_loss = self.test_loss / (self.window_size * len(self.test_loader) + 1)

        # concat and convert to numpy array
        rotations_ba = np.concatenate(rotations_ba, axis=0)        # N x 3 x 3
        translations_ba = np.concatenate(translations_ba, axis=0)  # N x 3 x 1
        rotations_ba_pred = np.concatenate(rotations_ba_pred, axis=0)
        translations_ba_pred = np.concatenate(translations_ba_pred, axis=0)
        transforms_ia = np.concatenate(transforms_ia, axis=0)      # N x 4 x 4
        transforms_ib = np.concatenate(transforms_ib, axis=0)

        eulers_ba = np.concatenate(eulers_ba, axis=0)  # N x 3
        eulers_ba_pred = np.concatenate(eulers_ba_pred, axis=0)

        test_r_mse_ba = np.mean((eulers_ba_pred - eulers_ba) ** 2)
        test_r_rmse_ba = np.sqrt(test_r_mse_ba)
        test_r_mae_ba = np.mean(np.abs(eulers_ba_pred - eulers_ba))
        test_t_mse_ba = np.mean((translations_ba - translations_ba_pred) ** 2)
        test_t_rmse_ba = np.sqrt(test_t_mse_ba)
        test_t_mae_ba = np.mean(np.abs(translations_ba - translations_ba_pred))

        self.textio.cprint('==FINAL TEST==')
        self.textio.cprint('A--------->B')
        self.textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (-1, self.test_loss, test_r_mse_ba, test_r_rmse_ba, test_r_mae_ba,
                         test_t_mse_ba, test_t_rmse_ba, test_t_mae_ba))

        ######################################################################
        ############## PLOT TRAJECTORY ##############
        ######################################################################
        # Compound poses and visualize the trajectory
        # test_translations_ab: Nx3
        # test_rotations_ab: Nx3x3
        # poses = np.array(dataset.poses)
        # translations_ab = poses[:,:3,3]
        # translations_ab -= translations_ab[0,:]

        # Compound poses
        T_ia_pred_list = [np.identity(4), ]
        T_ib_pred = np.identity(4)
        T_ba_pred = np.identity(4)
        T_ia_list = [np.identity(4), ]
        T_ia_list_extra = [np.identity(4), ]
        T_ib = np.identity(4)
        T_ba = np.identity(4)
        for item in range(rotations_ba_pred.shape[0]):
            # compound predictions
            R_ba_np_pred = rotations_ba_pred[item, :, :]
            translation_ba_np_pred = translations_ba_pred[item, :, 0] # Technically this is r_a_b_inb
            T_ba_pred[:3, :3] = R_ba_np_pred
            T_ba_pred[:3, 3] = translation_ba_np_pred

            T_ia_pred = T_ib_pred @ T_ba_pred
            T_ib_pred = T_ia_pred
            T_ia_pred_list.append(-T_ia_pred)

            # extract ground truth
            T_ia_np = transforms_ia[item, :, :]
            T_ia_list.append(T_ia_np)

        # this is only for evaluation purpose (plot_route)
        T_ia_np_pred = np.array(T_ia_pred_list)
        abs_translations_ia_pred = T_ia_np_pred[:,:3,3] # This is r_a_i_ini

        # same transform is applied to ground truth poses to revert from velodyne to camera frame
        T_ia_np = np.array(T_ia_list)
        abs_translations_ia = T_ia_np[:,:3,3]

        # save to file
        save_dict = {
            'R_ba_gt': rotations_ba,
            'R_ba_pred': rotations_ba_pred,
            'translation_ba_gt': translations_ba,
            'translation_ba_pred': translations_ba_pred,
            'translation_ib_gt': abs_translations_ia,
            'translation_ib_pred': abs_translations_ia_pred,
        }
        save_fname = '{}/pose_pred.pickle'.format(self.result_path)
        with open(save_fname, 'wb') as f:
            pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('======Save pose predictions to disk=======')

        test_eulers_ba = eulers_ba
        test_eulers_ba_pred = eulers_ba_pred

        # plot one color
        plt.clf()
        assert abs_translations_ia.shape[0] == abs_translations_ia_pred.shape[0]
        plot_route(abs_translations_ia, abs_translations_ia_pred, self.plot_path)

        print('======Plot trajectories to disk=======')

        # plot errors along each DOF
        # save translation plots
        save_name = '{}/abs_translation_error.png'.format(self.plot_path)
        plot_error(abs_translations_ia, abs_translations_ia_pred,
                   titles=['x (right)', 'y (down)', 'z (forward)'], setting='abs error', save_name=save_name)

        save_name = '{}/rel_translation_error.png'.format(self.plot_path)
        plot_error(translations_ba, translations_ba_pred,
                   titles=['x (right)', 'y (down)', 'z (forward)'], setting='error', save_name=save_name)

        save_name = '{}/rel_translation.png'.format(self.plot_path)
        plot_versus(translations_ba, translations_ba_pred,
                    titles=['x (right)', 'y (down)', 'z (forward)'], setting='error', save_name=save_name)

        # save rotation plots
        save_name = '{}/rel_rotation_error.png'.format(self.plot_path)
        plot_error(test_eulers_ba, test_eulers_ba_pred,
                   titles=['rot_x (pitch)', 'rot_y (yaw)', 'rot_z (roll)'], setting='error', save_name=save_name)

        save_name = '{}/rel_rotation.png'.format(self.plot_path)
        plot_versus(test_eulers_ba, test_eulers_ba_pred,
                    titles=['rot_x (pitch)', 'rot_y (yaw)', 'rot_z (roll)'], setting='error', save_name=save_name)
        print('======Plot errors along DOF to disk=======')

        return loss

    def test(self):
        """
        Full testing loop
        """

        self.test_epoch()