import numpy as np
import cv2
from scipy import ndimage

def cen2018features(fft_data: np.ndarray, min_range=58, zq=3.0, sigma_gauss=17,
                    max_range=1920, downsample=1) -> np.ndarray:
    """Extract features from polar radar data using the method described in cen_icra18
    Args:
        fft_data (np.ndarray): Polar radar power readings
        min_range (int): targets with a range bin less than or equal to this value will be ignored.
        zq (float): if y[i] > zq * sigma_q then it is considered a potential target point
        sigma_gauss (int): std dev of the gaussian filter used to smooth the radar signal
        
    Returns:
        np.ndarray: N x 2 array of feature locations (azimuth_bin, range_bin)
    """
    nazimuths = fft_data.shape[0]
    fft_data = fft_data[:, :max_range]
    fft_data = fft_data[:, ::downsample]
    sigma_gauss = int(sigma_gauss / downsample)
    if sigma_gauss % 2 == 0:
        sigma_gauss += 1
    # w_median = 200
    # q = fft_data - ndimage.median_filter(fft_data, size=(1, w_median))  # N x R
    q = fft_data - np.mean(fft_data, axis=1, keepdims=True)
    p = ndimage.gaussian_filter1d(q, sigma=sigma_gauss, truncate=3.0) # N x R
    noise = np.where(q < 0, q, 0) # N x R
    nonzero = np.sum(q < 0, axis=-1, keepdims=True) # N x 1
    sigma_q = np.sqrt(np.sum(noise**2, axis=-1, keepdims=True) / nonzero) # N x 1

    def norm(x, sigma):
        return np.exp(-0.5 * (x / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

    nqp = norm(q - p, sigma_q)
    npp = norm(p, sigma_q)
    nzero = norm(np.zeros((nazimuths, 1)), sigma_q)
    y = q * (1 - nqp / nzero) + p * ((nqp - npp) / nzero)
    t = np.nonzero(y > zq * sigma_q)
    # Extract peak centers
    current_azimuth = t[0][0]
    peak_points = [t[1][0]]
    peak_centers = []

    def mid_point(l):
        return l[len(l) // 2]

    for i in range(1, len(t[0])):
        if t[1][i] - peak_points[-1] > 1 or t[0][i] != current_azimuth:
            m = mid_point(peak_points)
            if m > min_range:
                peak_centers.append((current_azimuth, m))
            peak_points = []
        current_azimuth = t[0][i]
        peak_points.append(t[1][i])
    if len(peak_points) > 0 and mid_point(peak_points) > min_range:
        peak_centers.append((current_azimuth, mid_point(peak_points)))

    targets = np.asarray(peak_centers)
    targets[:, 1] *= downsample
    return np.asarray(peak_centers)

def polar_to_cartesian_points(azimuths: np.ndarray, polar_points: np.ndarray, radar_resolution: float,
    downsample_rate=1) -> np.ndarray:
    """Converts points from polar coordinates to cartesian coordinates
    Args:
        azimuths (np.ndarray): The actual azimuth of reach row in the fft data reported by the Navtech sensor
        polar_points (np.ndarray): N x 2 array of points (azimuth_bin, range_bin)
        radar_resolution (float): Resolution of the polar radar data (metres per pixel)
        downsample_rate (float): fft data may be downsampled along the range dimensions to speed up computation
    Returns:
        np.ndarray: N x 2 array of points (x, y) in metric
    """
    N = polar_points.shape[0]
    cart_points = np.zeros((N, 2))
    for i in range(0, N):
        azimuth = azimuths[int(polar_points[i, 0])]
        r = polar_points[i, 1] * radar_resolution * downsample_rate + radar_resolution / 2
        cart_points[i, 0] = r * np.cos(azimuth)
        cart_points[i, 1] = r * np.sin(azimuth)
    return cart_points

def convert_to_bev(cart_points: np.ndarray, cart_resolution: float, cart_pixel_width: int) -> np.ndarray:
    """Converts points from metric cartesian coordinates to pixel coordinates in the BEV image
    Args:
        cart_points (np.ndarray): N x 2 array of points (x, y) in metric
        cart_pixel_width (int): width and height of the output BEV image
    Returns:
        np.ndarray: N x 2 array of points (u, v) in pixels which can be plotted on the BEV image
    """
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    pixels = []
    N = cart_points.shape[0]
    for i in range(0, N):
        u = (cart_min_range + cart_points[i, 1]) / cart_resolution
        v = (cart_min_range - cart_points[i, 0]) / cart_resolution
        if 0 < u and u < cart_pixel_width and 0 < v and v < cart_pixel_width:
            pixels.append((u, v))
    return np.asarray(pixels)

def convert_to_bev2(cart_points, cart_resolution=0.2592, cart_pixel_width=640,
                    patch_size=21):
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    pixels = []
    targets = []
    N = cart_points.shape[0]
    for i in range(N):
        u = (cart_min_range + cart_points[i, 1]) / cart_resolution
        v = (cart_min_range - cart_points[i, 0]) / cart_resolution
        if 0 < u - patch_size and u + patch_size + 1 < cart_pixel_width and \
            0 < v - patch_size and v + patch_size + 1 < cart_pixel_width:
            pixels.append((u, v))
            targets.append((cart_points[i, 0], cart_points[i, 1]))
    return np.asarray(targets), np.asarray(pixels)

def convert_to_keypoints(targets, patch_size=21):
    N = targets.shape[0]
    kp = []
    for i in range(N):
        kp.append(cv2.KeyPoint(targets[i, 0], targets[i, 1], patch_size))
    return kp

