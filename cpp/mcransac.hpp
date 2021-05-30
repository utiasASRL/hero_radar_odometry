//////////////////////////////////////////////////////////////////////////////////////////////
/// \file mcransac.hpp
///
/// \author Keenan Burnett
/// \brief Rigid and motion-compensated RANSAC implementations along with some auxilliary
///     SE(3) math functions.
//////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <steam/steam.hpp>

namespace p = boost::python;
namespace np = boost::python::numpy;

/*!
   \brief Enforce orthogonality conditions on the given rotation matrix such that det(R) == 1 and R.tranpose() * R = I
   \param R The input rotation matrix either 2x2 or 3x3, will be overwritten with a slightly modified matrix to
   satisfy orthogonality conditions.
*/
void enforce_orthogonality(Eigen::MatrixXd &R);

/*!
   \brief Retrieve the rigid transformation that transforms points in p1 into points in p2.
   The output transform type (float or double) and size SE(2) vs. SE(3) depends on the size of the input points p1, p2.
   \param p1 A dim x N vector of points in either 2D (dim = 2) or 3D (dim = 3)
   \param p2 A dim x N vector of points in either 2D (dim = 2) or 3D (dim = 3)
   \param Tf [out] This matrix will be overwritten as the output transform
   \pre p1 and p2 are the same size. p1 and p2 are the matched feature point locations between two point clouds
   \post orthogonality is enforced on the rotation matrix.
*/
void get_rigid_transform(Eigen::MatrixXd p1, Eigen::MatrixXd p2, Eigen::MatrixXd &Tf);

/*!
   \brief Returns a random subset of indices, where 0 <= indices[i] <= max_index. indices are non-repeating.
*/
std::vector<int> random_subset(int max_index, int subset_size);

/*!
   \brief Returns the output of the carrot operator.
   For 3 x 1 input, carrot(x) * y is equivalent to cross_product(x, y)
   For 6 x 1 input, x = [rho, phi]^T. out = [carrot(phi), rho; 0 0 0 1]
   \param x Input vector which can be 3 x 1 or 6 x 1.
   \return If the input if 3 x 1, the output is 3 x 3, if the input is 6 x 1, the output is 4 x 4.
*/
Eigen::MatrixXd carrot(Eigen::VectorXd x);

/*!
   \brief Returns the output of the circledot operator. carrot(epsilon) * p == circledot(p) * epsilon,
   where epsilon is 6x1 and p is 4 x 1 homogeneous.
   p = [rhobar, eta]^T  circledot(p) = [eta * identity(3), -carrot(rhobar); 0 0 0 0 0 0]
   \param x Input is a 4 x 1 homogeneous 3D vector.
   \return returns the 4 x 6 output of circledot(x)
*/
Eigen::MatrixXd circledot(Eigen::VectorXd x);

/*!
   \brief This function converts from a lie vector to a 4 x 4 SE(3) transform.
   // Lie Vector xi = [rho, phi]^T (6 x 1) --> SE(3) T = [C, R; 0 0 0 1] (4 x 4)
   \param x Input vector is 6 x 1
   \return Output is 4 x SE(3) transform
*/
Eigen::Matrix4d se3ToSE3(Eigen::MatrixXd xi);

/*!
   \brief This function converts from an SE(3) transform into a lie vector
   // SE(3) T = [C, R; 0 0 0 1] (4 x 4) --> Lie Vector xi = [rho, phi]^T (6 x 1)
   \param T Input is a 4x4 SE(3) transform
   \return Output is 6x1 lie vector
*/
Eigen::VectorXd SE3tose3(Eigen::MatrixXd T);

/*!
   \brief Ensures that theta is within [0, 2 * pi)
*/
double wrapto2pi(double theta);

//* Ransac
/**
* \brief This class estimates a single rigid transform between two point clouds using RANSAC and singular value decomp
*/
class Ransac {
public:
    // p1, p2 need to be either (x, y) x N or (x, y, z) x N (must be in homogeneous coordinates)
    Ransac(const np::ndarray& p1_, const np::ndarray& p2_) {
        uint dim = p1_.shape(1);
        uint N = p1_.shape(0);
        assert(N == p2_.shape(0) && dim == p2_.shape(1) && dim >= 2);
        if (dim > 3)
            dim = 3;
        p1 = Eigen::MatrixXd::Zero(dim, N);
        p2 = Eigen::MatrixXd::Zero(dim, N);
        for (uint i = 0; i < dim; ++i) {
            for (uint j = 0; j < N; ++j) {
                p1(i, j) = double(p::extract<float>(p1_[j][i]));
                p2(i, j) = double(p::extract<float>(p2_[j][i]));
            }
        }
        T_best = Eigen::MatrixXd::Identity(dim + 1, dim + 1);
    }
    void setTolerance(double tolerance_) {tolerance = tolerance_;}
    void setInlierRatio(double inlier_ratio_) {inlier_ratio = inlier_ratio_;}
    void setMaxIterations(int iterations_) {iterations = iterations_;}
    void getTransform(Eigen::MatrixXd &Tf) {Tf = T_best;}

    /*!
       \brief Computes the transform that best aligns the two pointclouds such at T * p1 = p2
    */
    double computeModel();

    /*!
       \brief Retrieves the set of point pairs which are inliers given the current transform Tf.
    */
    void getInliers(Eigen::MatrixXd Tf, std::vector<int> &inliers);

private:
    Eigen::MatrixXd p1, p2;
    double tolerance = 0.35;
    double inlier_ratio = 0.9;
    int iterations = 100;
    Eigen::MatrixXd T_best;
};

//* MCRansac
/**
* \brief This class estimates the linear velocity and angular velocity of the sensor in the body-frame.
*
* Assuming constant velocity, the motion vector can be used to estimate the transform between any two pairs of points
* if the delta_t between those points issrand(t1_[i-1][0]); known.
*
* A single transform between the two pointclouds can also be retrieved.
*
* All operations are done in SE(3) even if the input is 2D. The output motion and transforms are in 3D.
*/
class MCRansac {
public:
    MCRansac(const np::ndarray& p1_, const np::ndarray& p2_, const np::ndarray& t1_, const np::ndarray& t2_) {
        assert(p1_.shape(0) == p2_.shape(0) && p1_.shape(1) == p2_.shape(1) && p1_.shape(0) >= p1_.shape(1));
        uint N = p1_.shape(0);
        uint dim = p1_.shape(1);
        if (dim > 3)
            dim = 3;
        p1bar = Eigen::MatrixXd::Zero(4, N);
        p2bar = Eigen::MatrixXd::Zero(4, N);
        p1bar.block(3, 0, 1, N) = Eigen::MatrixXd::Ones(1, N);
        p2bar.block(3, 0, 1, N) = Eigen::MatrixXd::Ones(1, N);
        for (uint i = 0; i < dim; ++i) {
            for (uint j = 0; j < N; ++j) {
                p1bar(i, j) = double(p::extract<float>(p1_[j][i]));
                p2bar(i, j) = double(p::extract<float>(p2_[j][i]));
            }
        }
        std::vector<int64_t> t1(N, 0);
        std::vector<int64_t> t2(N, 0);
        for (uint i = 0; i < N; ++i) {
            t1[i] = int64_t(p::extract<int64_t>(t1_[i]));
            t2[i] = int64_t(p::extract<int64_t>(t2_[i]));
        }
        R_pol << pow(0.25, 2), 0, 0, 0, 0, pow(0.0157, 2), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
        delta_ts = std::vector<double>(N, 0.0);
        for (uint i = 0; i < N; ++i) {
            int64_t delta_t = t2[i] - t1[i];
            delta_ts[i] = double(delta_t) / 1000000.0;
            if (delta_ts[i] > max_delta_t) {
                max_delta_t = delta_ts[i];
            }
            if (delta_ts[i] < min_delta_t) {
                min_delta_t = delta_ts[i];
            }
        }
        double delta_diff = (max_delta_t - min_delta_t) / (num_transforms - 1);
        for (int i = 0; i < num_transforms; ++i) {
            delta_vec.push_back(min_delta_t + i * delta_diff);
        }
    }
    void setTolerance(double tolerance_) {tolerance = tolerance_;}
    void setInlierRatio(double inlier_ratio_) {inlier_ratio = inlier_ratio_;}
    void setMaxIterations(int iterations_) {iterations = iterations_;}
    void setMaxGNIterations(int iterations_) {max_gn_iterations = iterations_;}
    void setConvergenceThreshold(double eps) {epsilon_converge = eps;}
    void correctForDoppler(bool doppler_) {doppler = doppler_;}
    void getTransform(double delta_t, Eigen::MatrixXd &Tf);
    void getMotion(Eigen::VectorXd &w) {w = w_best;}
    void setDopplerParameter(double beta_) {beta = beta_;}

    /*!
       \brief Computes the ego-motion vector that best aligns the two pointclouds
    */
    double computeModel();

    /*!
       \brief Retrieves the set of point pairs which are inliers given the current motion estimate.
    */
    void getInliers(Eigen::VectorXd wbar, std::vector<int> &inliers);

private:
    Eigen::MatrixXd p1bar, p2bar;
    std::vector<double> delta_ts;
    double tolerance = 0.1225;
    double inlier_ratio = 0.9;
    int iterations = 100;
    int max_gn_iterations = 10;
    double epsilon_converge = 0.0001;
    double error_converge = 0.01;
    int dim = 2;
    double beta = -0.049;  // beta = (f_t / (df / dt))
    double r_observable_sq = 0.0625;
    bool doppler = false;
    int num_transforms = 21;
    double max_delta_t = 0.0;
    double min_delta_t = 0.5;
    std::vector<double> delta_vec;
    Eigen::VectorXd w_best = Eigen::VectorXd::Zero(6);
    Eigen::Matrix4d R_pol = Eigen::Matrix4d::Identity();

    /*!
       \brief Given two sets of point pairs (p1small, p2small), this function computes the motion of the sensor
       (linear and angular velocity) in the body frame using nonlinear least squares.
       \pre It's very important that the delt_t_local is accurate. Note that each azimuth in the radar scan is time
       stamped, this should be used to get the more accurate time differences.
    */
    void get_motion_parameters(std::vector<int> subset, Eigen::VectorXd &wbar);

    /*!
       \brief Retrieve the number of inliers corresponding to body motion vector wbar. (6 x 1)
    */
    int getNumInliers(Eigen::VectorXd wbar);

    /*!
       \brief Given a body motion vector wbar (6 x 1), adjust the position of point p to account
       for the Doppler distortion which may be present in the data.
    */
    void dopplerCorrection(Eigen::VectorXd wbar, Eigen::VectorXd &p);
};

// Return the inverse of a 4x4 homogeneous transformation matrix
Eigen::Matrix4d get_inverse_tf(Eigen::Matrix4d T);
