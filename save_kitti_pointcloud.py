import pykitti
import argparse
from evtk.hl import pointsToVTK
import numpy as np
import json
from utils.helper_func import *

parser = argparse.ArgumentParser(description='Point Cloud Registration')
parser.add_argument('--config', default=None, type=str,
                    help='config file path (default: None)')
parser.add_argument('--save_dir', type=str, default='.', metavar='N',
                    help='Save to current directory by default')
parser.add_argument('--sequence_number', type=str, default='00', metavar='N',
                    help="Sequence number. String, e.g '00'")
# parser.add_argument('--num_frames', type=int, default=10, metavar='N',
#                     help='Number of frames to write. Integer value: e.g 10. Set to -1 for all frames.')

args = parser.parse_args()

with open(args.config) as f:
    config = json.load(f)

save_dir = os.path.join(args.save_dir, args.sequence_number)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
dataset = pykitti.odometry(config["dataset"]["data_dir"], args.sequence_number)

# take point cloud
num_poses = len(dataset.poses)
sample_n = num_poses
sample_idx = np.arange(sample_n)

# load image configs
azi_res = config["dataset"]["images"]["azi_res"]
azi_min = config["dataset"]["images"]["azi_min"]
azi_max = config["dataset"]["images"]["azi_max"]
ele_res = config["dataset"]["images"]["ele_res"]
ele_min = config["dataset"]["images"]["ele_min"]
ele_max = config["dataset"]["images"]["ele_max"]
input_channel = ['vertex', 'intensity', 'range'] # fix input to all channels

# image sizes
horizontal_pix = np.int32((azi_max - azi_min) / azi_res)
vertical_pix = np.int32((ele_max - ele_min) / ele_res)
geometry_img = np.zeros((3, vertical_pix, horizontal_pix), dtype=np.float32)

for i in sample_idx:

    points = dataset.get_velo(i)

    x = np.asarray(points[:,0], order='C')
    y = np.asarray(points[:,1], order='C')
    z = np.asarray(points[:,2], order='C')
    intensity = np.asarray(points[:,3], order='C')
    pointsToVTK("{}/raw_{}".format(save_dir, i), x, y, z,
                data = {"intensity" : intensity})

    geometry_img, input_img = pc2img_laser_to_row(points, azi_res, azi_min, azi_max,
                                                  input_channel, horizontal_pix)
    image_pc = np.reshape(geometry_img, (geometry_img.shape[0], -1))
    x = np.asarray(image_pc[0,:], order='C')
    y = np.asarray(image_pc[1,:], order='C')
    z = np.asarray(image_pc[2,:], order='C')
    pointsToVTK("{}/img_{}".format(save_dir, i), x, y, z)

    print(i)

print("====================")
print("===WRITE COMPLETE===")
print("====================")