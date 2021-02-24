import numpy as np
import cv2
import torch
from utils.utils import get_transform, get_inverse_tf
from datasets.radar import radar_polar_to_cartesian
import matplotlib.pyplot as plt

# apply a random transformation to every other frame and adjust the ground truth transform accordingly
def augmentBatch(batch, config):
    rot_max = config['augmentation']['rot_max']
    data = batch['data'].numpy()    # this seems to return a reference, not a copy
    # data2 = batch['data'].numpy().copy()    # this seems to return a reference, not a copy
    mask = batch['mask'].numpy()
    # mask2 = batch['mask'].numpy().copy()
    times_img = batch['times_img'].numpy()
    # times_img2 = batch['times_img'].numpy().copy()
    azimuths = batch['azimuths'].numpy()
    azimuth_times = batch['azimuth_times'].numpy()
    polar = batch['polar'].numpy()
    polar_mask = batch['polar_mask'].numpy()

    B, C, H, W = data.shape
    T_aug = []
    for i in range(B):
        if np.mod(i, config['window_size']) == 0:
            continue
        # img = data[i].squeeze()
        # mmg = mask[i].squeeze()
        # tmg = times_img[i].squeeze()
        rot = np.random.uniform(-rot_max, rot_max)
        # M = cv2.getRotationMatrix2D((W / 2, H / 2), rot * 180 / np.pi, 1.0)
        # data2[i] = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC).reshape(C, H, W)
        # mask2[i] = cv2.warpAffine(mmg, M, (W, H), flags=cv2.INTER_CUBIC).reshape(C, H, W)
        # times_img[i] = cv2.warpAffine(tmg, M, (W, H), flags=cv2.INTER_CUBIC).reshape(C, H, W)
        # azimuths[i] - rot

        azimuth_step = float(azimuths[i, 1] - azimuths[i, 0])
        index_steps = int(np.fabs(rot) // azimuth_step)
        azimuth_times[i] = np.roll(azimuth_times[i], -int(np.sign(rot))*index_steps, axis=0)
        times_img[i] = radar_polar_to_cartesian(azimuths[i], azimuth_times[i],
                                             config['radar_resolution'],
                                             config['cart_resolution'],
                                             config['cart_pixel_width']).astype(np.float32)

        polar[i] = np.roll(polar[i], -int(np.sign(rot))*index_steps, axis=0)
        data[i] = radar_polar_to_cartesian(azimuths[i], polar[i],
                                             config['radar_resolution'],
                                             config['cart_resolution'],
                                             config['cart_pixel_width']).astype(np.float32)

        polar_mask[i] = np.roll(polar_mask[i], -int(np.sign(rot))*index_steps, axis=0)
        mask[i] = radar_polar_to_cartesian(azimuths[i], polar_mask[i],
                                             config['radar_resolution'],
                                             config['cart_resolution'],
                                             config['cart_pixel_width']).astype(np.float32)

        # fig, ax = plt.subplots(3, 2)
        # im1 = ax[0, 0].imshow(times_img[i, 0])
        # im2 = ax[0, 1].imshow(times_img2[i, 0])
        # ax[1, 0].imshow(data[i, 0])
        # ax[1, 1].imshow(data2[i, 0])
        # ax[2, 0].imshow(mask[i, 0])
        # ax[2, 1].imshow(mask2[i, 0])
        # fig.colorbar(im1, ax=ax[0, 0])
        # fig.colorbar(im2, ax=ax[0, 1])
        # plt.show()

        # T_aug += [torch.from_numpy(get_transform(0, 0, -rot))]
        T_aug += [torch.from_numpy(get_transform(0, 0, azimuth_step*index_steps))]
    batch['data'] = torch.from_numpy(data)
    batch['times_img'] = torch.from_numpy(times_img)
    batch['mask'] = torch.from_numpy(mask > 0.5).type(batch['data'].dtype)    # make into a binary mask
    batch['T_aug'] = T_aug
    return batch
