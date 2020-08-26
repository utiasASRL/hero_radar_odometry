# sys imports
import os
import random
import sys

# third-party imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from PIL import Image
from torchvision import transforms

# project imports
from utils.utils import T_inv

def pc2img(pc, geometry_img, azi_res, azi_min, azi_max,
           ele_res, ele_min, ele_max, input_channel,
           horizontal_pix, vertical_pix):

    # sort the points based on range
    pc_xyz = pc[:,:3]
    r = np.sqrt(np.sum(pc_xyz ** 2, axis=1))
    order = np.argsort(r)
    pc_xyz = pc_xyz[order[::-1]]

    # compute azimuth and elevation
    azimuth = np.rad2deg(np.arctan2(pc_xyz[:,1], pc_xyz[:,0]))

    xy = np.sqrt(pc_xyz[:,0] ** 2 + pc_xyz[:,1] ** 2)
    elevation = np.rad2deg(np.arctan2(pc_xyz[:,2], xy))

    # reject points outside the field of view
    ids = (azimuth >= azi_min) * (azimuth <= azi_max) * (elevation >= ele_min) * (elevation <= ele_max)

    # compute u,v
    u = np.int32((0.5 * (azi_max - azi_min) - azimuth[ids]) // azi_res)
    u[u == horizontal_pix] = 0
    v = np.int32((ele_max - elevation[ids]) // ele_res)
    v[v == vertical_pix] = 0

    # assign to geometry image
    geometry_img[0, v, u] = pc_xyz[ids][:,0]
    geometry_img[1, v, u] = pc_xyz[ids][:,1]
    geometry_img[2, v, u] = pc_xyz[ids][:,2]

    # parse input channels
    input_images = []
    if "vertex" in input_channel:
        vertex_img = geometry_img.copy()
        input_images.append(vertex_img)
    if "intensity" in input_channel:
        intensity_img = np.zeros((1, vertical_pix, horizontal_pix), dtype=np.float32)
        pc_i = pc[:,3:]
        pc_i = pc_i[order[::-1]]
        intensity_img[0, v, u] = pc_i[ids][:,0]
        input_images.append(intensity_img)
    if "range" in input_channel:
        range_img = np.sqrt(np.sum(geometry_img ** 2, axis=0, keepdims=True))
        input_images.append(range_img)

    input_img = np.vstack(input_images)

    return geometry_img, input_img


def pc2img_laser_to_row(pc, azi_res, azi_min, azi_max, input_channel, horizontal_pix):
    '''
    Convert a point cloud to a LiDAR vertex image, where each laser is mapped to a row
    :param pc: point cloud Nx4
    :return: vertex image and (range image)
    '''

    # create image
    vertical_pix = 64
    image = np.zeros((5, vertical_pix, horizontal_pix), dtype=np.float32)

    # compute range
    pc_xyz = pc[:,:3]
    intensity = pc[:,3]
    range_m = np.sqrt(np.sum(pc_xyz ** 2, axis=1))

    # compute azimuth and elevation
    azimuth = np.rad2deg(np.arctan2(pc_xyz[:,1], pc_xyz[:,0]))
    xy = np.sqrt(pc_xyz[:,0] ** 2 + pc_xyz[:,1] ** 2)
    elevation = np.rad2deg(np.arctan2(pc_xyz[:,2], xy))

    # partition into lasers
    start_id = [0]
    end_id = []
    counter = 50 # start later than 0 in case first azimuth is 0
    while counter < pc.shape[0] - 1:
        # move to next
        counter += 1

        # check for 0 crossing
        if azimuth[counter-1] <= 0 and azimuth[counter] > 0 and abs(azimuth[counter]) < 100:
            end_id += [counter-1]
            start_id += [counter]

    end_id += [pc.shape[0] - 1]

    for l, _ in enumerate(start_id):
        a = start_id[l]
        b = end_id[l] + 1

        # entries for single laser
        l_xyz = pc_xyz[a:b, :3]
        l_azimuth = azimuth[a:b]
        l_intensity = intensity[a:b]
        l_range = range_m[a:b]

        # sort for range order
        order = np.argsort(l_range)
        # l_xyz = l_xyz[order[::-1]]
        l_azimuth = l_azimuth[order[::-1]]
        # l_intensity = l_intensity[order[::-1]]
        # l_range = l_range[order[::-1]]

        u = np.int32((0.5 * (azi_max - azi_min) - l_azimuth) // azi_res)
        u[u == horizontal_pix] = 0
        image[0, l, u] = l_xyz[order[::-1], 0]  # x
        image[1, l, u] = l_xyz[order[::-1], 1]  # y
        image[2, l, u] = l_xyz[order[::-1], 2]  # z
        image[3, l, u] = l_intensity[order[::-1]]  # intensity
        image[4, l, u] = l_range[order[::-1]]  # range

    # parse input channels
    input_images = []
    if "vertex" in input_channel:
        input_images.append(image[:3, :, :])
    if "intensity" in input_channel:
        input_images.append(image[3:4, :, :])
    if "range" in input_channel:
        input_images.append(image[4:5, :, :])

    input_img = np.vstack(input_images)
    geometry_img = image[:3, :, :]

    return geometry_img, input_img


def load_lidar_image(pc, config, rand_T=None, debug=False):
    '''
    Convert a point cloud to a LiDAR vertex image
    :param pc: point cloud Nx4
    :param config:
    :param debug: visualize the range image for debugging purpose. Default is False
    :return: vertex image and (range image)
    '''
    # load image configs
    azi_res = config["dataset"]["images"]["azi_res"]
    azi_min = config["dataset"]["images"]["azi_min"]
    azi_max = config["dataset"]["images"]["azi_max"]
    ele_res = config["dataset"]["images"]["ele_res"]
    ele_min = config["dataset"]["images"]["ele_min"]
    ele_max = config["dataset"]["images"]["ele_max"]
    input_channel = config["dataset"]["images"]["input_channel"]

    # image sizes
    horizontal_pix = np.int32((azi_max - azi_min) / azi_res)
    vertical_pix = np.int32((ele_max - ele_min) / ele_res)
    geometry_img = np.zeros((3, vertical_pix, horizontal_pix), dtype=np.float32)

    if rand_T is not None:
        assert rand_T.shape == (4, 4), "Transformation matrix must be 4x4!"

        num_pts = pc.shape[0]
        pc_aug = np.hstack((pc[:,:3], np.ones((num_pts, 1))))
        pc_aug = (rand_T @ pc_aug.T).T
        pc[:,:3] = pc_aug[:,:3]

    if config["dataset"]["images"]["map_laser_to_row"]:
        geometry_img, input_img = pc2img_laser_to_row(pc, azi_res, azi_min, azi_max,
                                                      input_channel, horizontal_pix)
    else:
        geometry_img, input_img = pc2img(pc, geometry_img, azi_res, azi_min, azi_max,
                                        ele_res, ele_min, ele_max, input_channel,
                                        horizontal_pix, vertical_pix)

    return geometry_img, input_img


def pc2img_slow(pc, config, rand_T=None):
    # load image configs
    azi_res = config.azi_res
    azi_min = config.azi_min
    azi_max = config.azi_max
    ele_res = config.ele_res
    ele_min = config.ele_min
    ele_max = config.ele_max
    
    horizontal_pix = np.int32((azi_max - azi_min) / azi_res)
    vertical_pix = np.int32((ele_max - ele_min) / ele_res)
    num_channel = 3
    vertex_img = np.zeros((vertical_pix, horizontal_pix, num_channel))
    popu_img = np.ones((vertical_pix, horizontal_pix)) * 1e4
    last_used = np.zeros((vertical_pix, horizontal_pix), dtype=np.int32)
    
    if rand_T is not None:
        assert rand_T.shape == (4, 4), "Transformation matrix must be 4x4!"

        num_pts = pc.shape[0]
        pc_aug = np.hstack((pc[:,:3], np.ones((num_pts, 1))))
        pc_aug = (rand_T @ pc_aug.T).T
        pc[:,:3] = pc_aug[:,:3]

    pc_xyz = pc[:,:3]

    # compute azimuth and elevation
    azimuth = np.rad2deg(np.arctan2(pc_xyz[:,1], pc_xyz[:,0]))

    xy = np.sqrt(pc_xyz[:,0] ** 2 + pc_xyz[:,1] ** 2)
    elevation = np.rad2deg(np.arctan2(pc_xyz[:,2], xy))

    # reject points outside the field of view
    ids = (azimuth >= azi_min) * (azimuth <= azi_max) * (elevation >= ele_min) * (elevation <= ele_max)

    # compute u,v
    u = np.int32((0.5 * (azi_max - azi_min) - azimuth[ids]) // azi_res)
    u[u == horizontal_pix] = 0
    v = np.int32((ele_max - elevation[ids]) // ele_res)

    # compute range
    r = np.sqrt(np.sum(pc_xyz[ids] ** 2, axis=1))

    # populate image with nearest point
    pt_used = np.zeros(np.count_nonzero(ids))
    ignore_num = 0
    for i in range(u.shape[0]):
        if r[i] < popu_img[v[i], u[i]]:
            if popu_img[v[i], u[i]] != 1e4:
                ignore_num += 1
                pt_used[last_used[v[i], u[i]]] = 0
            last_used[v[i], u[i]] = i
            pt_used[i] = 1
            popu_img[v[i], u[i]] = r[i]
            vertex_img[v[i], u[i], 0] = pc_xyz[ids][i,0]
            vertex_img[v[i], u[i], 1] = pc_xyz[ids][i,1]
            vertex_img[v[i], u[i], 2] = pc_xyz[ids][i,2]
        else:
            ignore_num += 1
    return vertex_img

def rad2deg(radian):
    return radian / torch.Tensor([math.pi]).cuda() * 180.0

def compute_2D_from_3D(points, config):
    '''
    Compute u, v 2D coordinates from 3D points
    :param points: Bx3xN
    :param config:
    :return: Bx2xN
    '''
    # load image configs
    azi_res = config["dataset"]["images"]["azi_res"]
    azi_min = config["dataset"]["images"]["azi_min"]
    azi_max = config["dataset"]["images"]["azi_max"]
    ele_res = config["dataset"]["images"]["ele_res"]
    ele_min = config["dataset"]["images"]["ele_min"]
    ele_max = config["dataset"]["images"]["ele_max"]

    # project to 2D
    horizontal_pix = np.int32((azi_max - azi_min) / azi_res)
    vertical_pix = np.int32((ele_max - ele_min) / ele_res)
    azimuth = rad2deg(torch.atan2(points[:,1,:], points[:,0,:]))
    xy = torch.sqrt(points[:,0,:] ** 2 + points[:,1,:] ** 2)
    elevation = rad2deg(torch.atan2(points[:,2,:], xy))
    u = ((0.5 * (azi_max - azi_min) - azimuth) / azi_res)
    u[u == horizontal_pix] = 0
    v = ((ele_max - elevation) / ele_res)
    v[v == vertical_pix] = 0
    points_2D = torch.cat((u.unsqueeze(-1), v.unsqueeze(-1)), dim=2)

    return points_2D

def compute_disparity(left_img, right_img, config):

    # f * b ~= 388, closest dist ~= 388 / 96 = 4.04
    # The ground is farther than 4.04, but this captures
    # sign posts that mostly seem to be the closes objects.
    stereo = cv2.StereoSGBM_create(**config['dataset']['disparity'])

    disp = stereo.compute(left_img, right_img)
    disp = disp.astype(np.float32) / 16.0

    disp[np.abs(disp) < 1e-10] = 1e-10

    return disp

def load_camera_data(left_img_file, right_img_file, height, width, config):
    
    left_img = Image.open(left_img_file)
    right_img = Image.open(right_img_file)

    width_actual, height_actual = left_img.size # PIL image size 

    # Crop images so they are all of the same size.
    if (height_actual > height) or (width_actual > width):
        i = random.randint(0, height_actual - height) 
        j = random.randint(0, width_actual - width)

        left_img = transforms.functional.crop(left_img, i, j, height, width)
        right_img = transforms.functional.crop(right_img, i, j, height, width)
       
    left_img_uint8 = np.uint8(left_img.copy())
    right_img_uint8 = np.uint8(right_img.copy())
        
    disparity_img = compute_disparity(left_img_uint8, right_img_uint8, config)

    to_tensor = transforms.ToTensor()
    left_img = to_tensor(left_img).numpy()
    right_img = to_tensor(right_img).numpy()

    return np.vstack((left_img, right_img)), disparity_img
