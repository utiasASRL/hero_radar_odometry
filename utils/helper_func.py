# sys imports
import os
import sys

# third-party imports
import numpy as np

# project imports

def pc2img(pc, config, rand_T=None, debug=False):
    '''
    Convert a point cloud to a LiDAR vertex image
    :param pc: point cloud Nx4
    :param config:
    :param debug: visualize the range image for debugging purpose. Default is False
    :return: vertex image and (range image)
    '''
    # load image configs
    azi_res = config.azi_res
    azi_min = config.azi_min
    azi_max = config.azi_max
    ele_res = config.ele_res
    ele_min = config.ele_min
    ele_max = config.ele_max

    # image sizes
    horizontal_pix = np.int32((azi_max - azi_min) / azi_res)
    vertical_pix = np.int32((ele_max - ele_min) / ele_res)
    num_channel = 3
    vertex_img = np.zeros((vertical_pix, horizontal_pix, num_channel))

    if rand_T is not None:
        assert rand_T.shape == (4, 4), "Transformation matrix must be 4x4!"

        num_pts = pc.shape[0]
        pc_aug = np.hstack((pc[:,:3], np.ones((num_pts, 1))))
        pc_aug = (rand_T @ pc_aug.T).T
        pc[:,:3] = pc_aug[:,:3]

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

    # assign to image
    vertex_img[v, u, 0] = pc_xyz[ids][:,0]
    vertex_img[v, u, 1] = pc_xyz[ids][:,1]
    vertex_img[v, u, 2] = pc_xyz[ids][:,2]

    # create range image
    if debug:
        vertex_range = np.sqrt(np.sum(vertex_img ** 2, axis=2))
        return vertex_img, vertex_range

    return vertex_img

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