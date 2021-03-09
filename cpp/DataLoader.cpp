#include "DataLoader.hpp"
#include <math.h>
#include <omp.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// timestamps: N x 1 , azimuths: N x 1, fft_data: N x R need to be sized correctly by the python user.
void DataLoader::load_radar(const std::string path, np::ndarray& timestamps, np::ndarray& azimuths,
    np::ndarray& fft_data) {
    int encoder_size = 5600;
    cv::Mat raw_example_data = cv::imread(path, cv::IMREAD_GRAYSCALE);
    for (uint i = 0; i < num_azimuths; ++i) {
        uchar* byteArray = raw_example_data.ptr<uchar>(i);
        timestamps[i][0] = *((int64_t *)(byteArray));
        azimuths[i][0] = *((uint16_t *)(byteArray + 8)) * 2 * M_PI / double(encoder_size);
        for (uint j = 0; j < min_range; ++j) {
            fft_data[i][j] = 0;
        }
        for (uint j = min_range; j < range_bins; ++j) {
            fft_data[i][j] = (float)*(byteArray + 11 + j) / 255.0;
        }
    }
}

static double get_azimuth_index(std::vector<float>& azimuths, double azimuth) {
    double mind = 1000;
    double closest = 0;
    for (uint i = 0; i < azimuths.size(); ++i) {
        double d = fabs(azimuths[i] - azimuth);
        if (d < mind) {
            mind = d;
            closest = i;
        }
    }
    if (azimuths[closest] < azimuth) {
        double delta = 0;
        if (closest < azimuths.size() - 1)
            delta = (azimuth - azimuths[closest]) / (azimuths[closest + 1] - azimuths[closest]);
        closest += delta;
    } else if (azimuths[closest] > azimuth){
        double delta = 0;
        if (closest > 0)
            delta = (azimuths[closest] - azimuth) / (azimuths[closest] - azimuths[closest - 1]);
        closest -= delta;
    }
    return closest;
}

// azimuths: N x 1, fft_data: N x R, cart: W x W need to be sized correctly by the python user.
void DataLoader::polar_to_cartesian(const np::ndarray& azimuths_np, const np::ndarray& fft_data, np::ndarray& cart) {
    std::vector<float> azimuths(num_azimuths, 0);
    for (uint i = 0; i < num_azimuths; ++i) {
        azimuths[i] = p::extract<float>(azimuths_np[i][0]);
    }

    cv::Mat polar = cv::Mat::zeros(num_azimuths, range_bins, CV_32F);
    for (uint i = 0; i < num_azimuths; ++i) {
        for (uint j = 0; j < range_bins; ++j) {
            polar.at<float>(i, j) = p::extract<float>(fft_data[i][j]);
        }
    }

    cv::Mat cart_img;
    float cart_min_range = (cart_pixel_width / 2) * cart_resolution;
    if (cart_pixel_width % 2 == 0)
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution;

    cv::Mat map_x = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);
    cv::Mat map_y = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);

    for (int j = 0; j < map_y.cols; ++j) {
        for (int i = 0; i < map_y.rows; ++i) {
            map_y.at<float>(i, j) = -1 * cart_min_range + j * cart_resolution;
        }
    }
    for (int i = 0; i < map_x.rows; ++i) {
        for (int j = 0; j < map_x.cols; ++j) {
            map_x.at<float>(i, j) = cart_min_range - i * cart_resolution;
        }
    }
    cv::Mat range = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);
    cv::Mat angle = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);

    double azimuth_step = azimuths[1] - azimuths[0];
    for (int i = 0; i < range.rows; ++i) {
        for (int j = 0; j < range.cols; ++j) {
            float x = map_x.at<float>(i, j);
            float y = map_y.at<float>(i, j);
            float r = (sqrt(pow(x, 2) + pow(y, 2)) - radar_resolution / 2) / radar_resolution;
            if (r < 0)
                r = 0;
            range.at<float>(i, j) = r;
            float theta = atan2f(y, x);
            if (theta < 0)
                theta += 2 * M_PI;
            if (navtech_version == CIR204) {
                angle.at<float>(i, j) = get_azimuth_index(azimuths, theta);
            } else {
                angle.at<float>(i, j) = (theta - azimuths[0]) / azimuth_step;
            }
        }
    }

    if (interpolate_crossover) {
        cv::Mat a0 = cv::Mat::zeros(1, range_bins, CV_32F);
        cv::Mat aN_1 = cv::Mat::zeros(1, range_bins, CV_32F);
        for (uint j = 0; j < range_bins; ++j) {
            a0.at<float>(0, j) = polar.at<float>(0, j);
            aN_1.at<float>(0, j) = polar.at<float>(num_azimuths-1, j);
        }
        cv::vconcat(aN_1, polar, polar);
        cv::vconcat(polar, a0, polar);
        angle = angle + 1;
    }
    cv::remap(polar, cart_img, range, angle, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));


    for (uint i = 0; i < cart_pixel_width; ++i) {
        for (uint j = 0; j < cart_pixel_width; ++j) {
            cart[i][j] = cart_img.at<float>(i, j);
        }
    }
}
