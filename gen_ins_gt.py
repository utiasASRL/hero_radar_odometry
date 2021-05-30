import os
import json
import numpy as np
from datasets.interpolate_poses import interpolate_ins_poses, so3_to_euler
from utils.utils import get_inverse_tf

def parse(line):
    line = line.split(',')
    out = [int(line[0]), int(line[1])]
    for i in range(2, 8):
        out.append(float(line[i]))
    out += [int(line[8]), int(line[9])]
    return out

if __name__ == '__main__':
    config = 'config/steam.json'
    with open(config) as f:
        config = json.load(f)

    T_radar_imu = np.identity(4, dtype=np.float32)
    for i in range(3):
        T_radar_imu[i, 3] = config['steam']['ex_translation_vs_in_s'][i]

    header = 'source_timestamp,destination_timestamp,x,y,z,roll,pitch,yaw,source_radar_timestamp,destination_radar_timestamp\n'
    gt_path = config['data_dir']
    seqs = [f for f in os.listdir(gt_path) if '2019' in f]
    seqs.sort()
    for seq in seqs:
        print('Extracting INS GT for: {}'.format(seq))
        with open(gt_path + seq + '/gt/radar_odometry.csv', 'r') as f:
            f.readline()
            odom_gt = f.readlines()
        f = open(gt_path + seq + '/gt/radar_odometry_ins.csv', 'w')
        f.write(header)

        gt_times = []
        for i in range(1, len(odom_gt)):
            gt_times.append(int(odom_gt[i].split(',')[1]))
        orig_time = int(odom_gt[0].split(',')[1])
        gt_times.append(int(odom_gt[-1].split(',')[0]))
        ins_path = gt_path + seq + '/gps/ins.csv'
        abs_poses = interpolate_ins_poses(ins_path, gt_times, orig_time)
        abs_poses.insert(0, np.identity(4, dtype=np.float32))
        for i in range(len(gt_times) - 1):
            T_0k = np.array(abs_poses[i])
            T_0kp1 = np.array(abs_poses[i + 1])
            T = get_inverse_tf(T_0k) @ T_0kp1  # T_k_kp1 (next time to current)
            T = T_radar_imu @ T @ get_inverse_tf(T_radar_imu)
            rpy = so3_to_euler(T[0:3, 0:3])
            phi = rpy[0, 2]
            odom = parse(odom_gt[i])
            f.write('{},{},{},{},{},{},{},{},{},{}\n'.format(odom[0], odom[1], T[0, 3], T[1, 3], 0, 0, 0, phi, odom[8], odom[9]))
