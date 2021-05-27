import unittest
import numpy as np
from utils.utils import getApproxTimeStamps, wrapto2pi

def get_times(t0, M=400):
	times = np.zeros(M, dtype=np.int64)
	delta_t = int(1e6 * 0.25 / M)
	for i in range(M):
		times[i] = int(t0 + i * delta_t)
	return times

def get_points(R, M=400):
	points = np.zeros((M, 2))
	azimuth_step = (2 * np.pi) / M
	for i in range(M):
		r = np.random.rand() * R
		points[i, 0] = r * np.cos(i * azimuth_step)
		points[i, 1] = r * np.sin(i * azimuth_step)
	return points

class TestTimeStamps(unittest.TestCase):
	def test_1(self):
		t0 = 1547131014000000
		R = 75
		times = get_times(t0)
		points = get_points(R)
		timesout = getApproxTimeStamps([points], [times])[0]
		for i in range(len(times)):
			self.assertTrue(abs(times[i] - timesout[i]) < 10)

	def test_2(self):
		N = 1000
		t0 = 1547131014000000
		M = 400
		points = np.random.randn(N, 2)
		times1 = np.zeros(N, dtype=np.int64)
		azimuth_step = (2 * np.pi) / M
		times = get_times(t0, M)
		times2 = getApproxTimeStamps([points], [times])[0]
		for i in range(N):
			azimuth = wrapto2pi(np.arctan2(points[i, 1], points[i, 0]))
			delta_t = int(1e6 * 0.25 * azimuth / (2 * np.pi))
			times1[i] = int(t0 + delta_t)
			self.assertTrue(abs(times1[i] - times2[i]) < 10, "{} != {}, {}, {}".format(times1[i], times2[i], azimuth, delta_t))

if __name__ == '__main__':
	unittest.main()
	