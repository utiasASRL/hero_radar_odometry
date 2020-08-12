import os
import sys
import time
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

from utils.lie_algebra import so3_to_rpy
from utils.helper_func import plot_route, plot_versus, plot_error

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
    Tester class
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

        checkpoint_path = "{}/{}/{}".format('results', config['previous_session'], 'checkpoints/chkp.tar')
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print("Model and training state restored.")

        # data loaders
        self.test_loader = test_loader

        # config dictionary
        self.config = config

        # logging
        self.result_path = os.path.join('results', config['session_name'])
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.log_path = os.path.join(self.result_path, 'eval.log')
        self.textio = IOStream(self.log_path)
        self.test_loss = 0

    def test_epoch(self):
        self.model.eval()

        # init saving lists
        rotations_ab = [] # store R_21
        translations_ab = [] # store t_21
        rotations_ab_pred = [] # store R_21_pred
        translations_ab_pred = [] # store t_21_pred

        eulers_ab = [] # store euler_12
        eulers_ab_pred = [] # store euler_12_pred

        with torch.no_grad():
            for i_batch, batch_sample in enumerate(self.test_loader):

                # forward prop
                loss = self.model(batch_sample)
                self.textio.cprint("loss:{}".format(loss['LOSS'].item()))
                self.test_loss += loss['LOSS'].item()

                # collect poses
                save_dict = self.model.return_save_dict()
                R_21_pred, t_21_pred = save_dict['R_pred'], save_dict['t_pred']
                R_21, t_21 = save_dict['R_tgt_src'], save_dict['t_tgt_src']
                euler_21 = so3_to_rpy(R_21)
                euler_21_pred = so3_to_rpy(R_21_pred)

                # append to list
                rotations_ab.append(R_21.detach().cpu().numpy())
                translations_ab.append(t_21.detach().cpu().numpy())
                rotations_ab_pred.append(R_21_pred.detach().cpu().numpy())
                translations_ab_pred.append(t_21_pred.detach().cpu().numpy())

                eulers_ab.append(euler_21.detach().cpu().numpy())
                eulers_ab_pred.append(euler_21_pred.detach().cpu().numpy())

            # concat and convert to numpy array
            rotations_ab = np.concatenate(rotations_ab, axis=0)
            translations_ab = np.concatenate(translations_ab, axis=0)
            rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
            translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

            eulers_ab = np.concatenate(eulers_ab, axis=0)
            eulers_ab_pred = np.concatenate(eulers_ab_pred, axis=0)

        test_r_mse_ab = np.mean((eulers_ab_pred - eulers_ab) ** 2)
        test_r_rmse_ab = np.sqrt(test_r_mse_ab)
        test_r_mae_ab = np.mean(np.abs(eulers_ab_pred - eulers_ab))
        test_t_mse_ab = np.mean((translations_ab - translations_ab_pred) ** 2)
        test_t_rmse_ab = np.sqrt(test_t_mse_ab)
        test_t_mae_ab = np.mean(np.abs(translations_ab - translations_ab_pred))

        self.textio.cprint('==FINAL TEST==')
        self.textio.cprint('A--------->B')
        self.textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (-1, self.test_loss, test_r_mse_ab, test_r_rmse_ab, test_r_mae_ab,
                         test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))

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
        Ts_ib_pred = [np.identity(4), ]
        T_ia_pred = np.identity(4)
        T_ab_pred = np.identity(4)
        Ts_ib = [np.identity(4), ]
        T_ia = np.identity(4)
        T_ab = np.identity(4)
        for item in range(rotations_ab_pred.shape[0]):
            # compound predictions
            R_ab_np_pred = rotations_ab_pred[item, :, :]
            translation_ab_np_pred = translations_ab_pred[item, :]
            T_ab_pred[:3, :3] = R_ab_np_pred
            T_ab_pred[:3, 3] = translation_ab_np_pred
            # T_ba_pred = np.linalg.inv(T_ab_pred)
            # T_ba_pred = dataset.T_cam0_velo[0] @ T_ba_pred @ dataset.T_velo_cam0[0]
            T_ib_pred = T_ia_pred @ T_ab_pred
            T_ia_pred = T_ib_pred
            Ts_ib_pred.append(T_ib_pred)

            # compound ground truth
            R_ab_np = rotations_ab[item, :, :]
            # print(test_rotations_ab)
            translation_ab_np = translations_ab[item, :]
            translation_ab_np = translation_ab_np
            T_ab[:3, :3] = R_ab_np
            T_ab[:3, 3] = translation_ab_np
            # T_ba = np.linalg.inv(T_ab)
            # T_ba = dataset.T_cam0_velo[0] @ T_ba @ dataset.T_velo_cam0[0]
            T_ib = T_ia @ T_ab
            T_ia = T_ib
            Ts_ib.append(T_ib)
            # print("T_ab:{}".format(T_ab))
            # print("R_ab_np:{}".format(R_ab_np))
            # print("translation_ab_np:{}".format(translation_ab_np))
        # transform the frame from velodyne to camera using calibration matrix in KITTI dataset
        # this is only for evaluation purpose (plot_route)
        Ts_ib_np_pred = np.array(Ts_ib_pred)
        # print(Ts_ib_np_pred.shape)
        # Ts_ib_np_pred = dataset.T_cam0_velo @ Ts_ib_np_pred @ dataset.T_velo_cam0
        translations_ab_pred = Ts_ib_np_pred[:,:3,3]
        # print("euler_ab_np_pred:{}".format(euler_ab_np_pred))
        # print("test_rotations_ab:{}".format(T_ab))
        # print("test_translations_ab:{}".format(test_translations_ab))

        # same transform is applied to ground truth poses to revert from velodyne to camera frame
        Ts_ib_np = np.array(Ts_ib)
        # Ts_ib_np = dataset.T_cam0_velo @ Ts_ib_np @ dataset.T_velo_cam0
        translations_ab = Ts_ib_np[:,:3,3]

        # scale up the translations
        # print("before:{}".format(translations_ab))
        # translations_ab = translations_ab * np.float32(args.scale)
        # print("after:{}".format(translations_ab))
        # translations_ab_pred = translations_ab_pred * np.float32(args.scale)

        # save to file
        save_dict = {
            'R_ab_gt': rotations_ab,
            'R_ab_pred': rotations_ab_pred,
            'translation_ab_gt': translations_ab,
            'translation_ab_pred': translations_ab_pred,
            'translation_gt': translations_ab,
            'translation_pred': translations_ab_pred,
        }
        save_fname = '{}/pose_pred.pickle'.format(self.result_path)
        with open(save_fname, 'wb') as f:
            pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('======Save pose predictions to disk=======')

        # plot one color
        assert translations_ab.shape[0] == translations_ab_pred.shape[0]
        plt.clf()
        plt.scatter([translations_ab[0][0]], [translations_ab[0][2]], label='sequence start', marker='s', color='k')
        plot_route(translations_ab, translations_ab_pred, 'r', 'b')
        plt.legend()
        plt.title('')
        save_dir = '{}/visualization'.format(self.result_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_name = '{}/route.png'.format(save_dir)
        plt.savefig(save_name)
        print('======Plot trajectories to disk=======')

        # plot errors along each DOF
        # print(test_eulers_ab.shape)
        test_eulers_ab = eulers_ab
        euler_ab_np_pred = eulers_ab_pred
        # print(test_eulers_ab.shape)

        # save translation plots
        save_name = '{}/abs_translation_error.png'.format(save_dir)
        plot_error(translations_ab, translations_ab_pred,
                   titles=['x', 'y', 'z'], setting='abs', save_name=save_name)

        save_name = '{}/rel_translation_error.png'.format(save_dir)
        plot_error(translations_ab, translations_ab_pred,
                   titles=['x', 'y', 'z'], setting='rel', save_name=save_name)

        save_name = '{}/rel_translation.png'.format(save_dir)
        plot_versus(translations_ab, translations_ab_pred,
                    titles=['x', 'y', 'z'], setting='rel', save_name=save_name)

        # save rotation plots
        print(test_eulers_ab.shape, euler_ab_np_pred.shape)
        save_name = '{}/rel_rotation_error.png'.format(save_dir)
        plot_error(test_eulers_ab, euler_ab_np_pred,
                   titles=['psi', 'theta', 'phi'], setting='rel', save_name=save_name)

        save_name = '{}/rel_rotation.png'.format(save_dir)
        plot_versus(test_eulers_ab, euler_ab_np_pred,
                    titles=['psi', 'theta', 'phi'], setting='rel', save_name=save_name)
        print('======Plot errors along DOF to disk=======')


        return loss

    def test(self):
        """
        Full testing loop
        """

        self.test_epoch()