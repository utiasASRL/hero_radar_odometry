import torch
import numpy as np
import cv2
from utils.utils import get_transform, get_inverse_tf

# apply a random transformation to every other frame and adjust the ground truth transform accordingly
def augmentBatch(batch, config):
    rot_max = config['augmentation']['rot_max']
    input = batch['input'].numpy()
    T_21 = batch['T_21'].numpy()
    B, C, H, W = input.shape
    for i in range(1, B, 2):
        img = input[i].squeeze()
        rot = np.random.uniform(-rot_max, rot_max)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rot * 180 / np.pi, 1.0)
        input[i] = cv2.warpAffine(img, M, (W, H), interpolation = cv2.INTER_CUBIC).reshape(C, H, W)
        T = get_transform(0, 0, rot)
        T_21[i - 1] = np.matmul(T, T_21[i - 1])
        T_21[i] = np.matmul(T_21[i], get_inverse_tf(T))
    batch['input'] = torch.from_numpy(input)
    batch['T_21'] = torch.from_numpy(T_21)
    return batch
