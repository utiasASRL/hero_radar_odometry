import pykitti
import argparse
from evtk.hl import pointsToVTK
import numpy as np

parser = argparse.ArgumentParser(description='Point Cloud Registration')
parser.add_argument('--save_dir', type=str, default='.', metavar='N',
                    help='Save to current directory by default')
parser.add_argument('--sequence_number', type=str, default='00', metavar='N',
                    help="Sequence number. String, e.g '00'")
parser.add_argument('--num_frames', type=int, default=10, metavar='N',
                    help='Number of frames to write. Integer value: e.g 10')

args = parser.parse_args()

base_dir = '/mnt/ssd1/research/dataset/KITTI/dataset'

save_dir = args.save_dir

dataset = pykitti.odometry(base_dir, args.sequence_number)

for i in range(args.num_frames):
    points = dataset.get_velo(i)

    x = np.asarray(points[:,0], order='C')
    y = np.asarray(points[:,1], order='C')
    z = np.asarray(points[:,2], order='C')
    intensity = np.asarray(points[:,3], order='C')
    pointsToVTK("{}/pointcloud{}".format(save_dir, i), x, y, z,
                data = {"intensity" : intensity})

print("====================")
print("===WRITE COMPLETE===")
print("====================")