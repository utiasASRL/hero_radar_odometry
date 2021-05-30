//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SteamSolver.hpp
///
/// \author David Yoon, Keenan Burnett
/// \brief A C++ class with a boost::python wrapper for optimizing odometry poses over a sliding
///     window with a motion prior.
//////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <vector>
#include <deque>
#include "SteamPyHelper.hpp"

class SteamSolver {
public:
    SteamSolver(const double& dt, const unsigned int& window_size) :
        dt_(dt), window_size_(window_size) {
        Eigen::Array<double, 1, 6> Qc_diag;
        Qc_diag << 0.37, 0.04, 0.13, 0.007, 0.008, 0.002;
        Qc_inv_.setZero();
        Qc_inv_.diagonal() = 1.0/Qc_diag;
        // Initialize extrinsic transform to identity
        T_sv_ = steam::se3::FixedTransformEvaluator::MakeShared(lgmath::se3::Transformation());
        // Initialize trajectory
        resetTraj();
    }

    /*!
       \brief Resets the trajectory to identity poses and zero velocities.
    */
    void resetTraj();

    /*!
       \brief Slides the window based on the previous estimation by popping the oldest pose and
        constraints, and motion prior.
    */
    void optimize();

    /*!
       \brief Retrieves the estimated 4x4 poses.
        Note: poses must be sized correctly on the Python side (W x 4 x 4)
    */
    void getPoses(np::ndarray& poses);

    /*!
       \brief Retrieves the estimated velocities.
        Note: vels must be sized correctly on the Python side (W x 6)
    */
    void getVelocities(np::ndarray& vels);

    void getSigmapoints2N(np::ndarray& sigma_T);

    /*!
       \brief Interpolates for the pose of the sensor between two given times: T_ba
        which transforms points in frame a into frame b. pose must be (4 x 4) on Python side.
    */
    void getPoseBetweenTimes(np::ndarray& pose, const int64_t ta, const int64_t tb);

    void setQcInv(const np::ndarray& Qc_diag);

    /*!
       \brief Loads in the measurements to be used in the STEAM optimization.
       \param p2_list List of numpy array (N, 3)
       \param p1_list List of numpy array (N, 3)
       \param weight_list List of numpy array (N, 3, 3)
       \param t2_list List of numpy array (N,) Timestamps for points in p2_list
       \param t1_list List of numpy array (N,) Timestamps for points in p1_list
       \param t_refs List of ints Reference times at which the poses will be estimated
    */
    void setMeas(const p::object& p2_list, const p::object& p1_list, const p::object& weight_list,
        const p::object& t2_list, const p::object& t1_list, const p::object& t_refs);
    void setExtrinsicTsv(const np::ndarray& T_sv);
    void setRansacVersion(const unsigned int& version) {ransac_version = uint(version);}
    void setZeroVelPriorFlag(const bool zero_vel) {zero_vel_prior_flag_ = zero_vel;}
    void setVelPriorFlag(const bool vel_prior) {vel_prior_ = vel_prior;}
    void useRansac() {use_ransac = true;}
    void useCTSteam() {ct_steam = true;}

private:
    // Solver
    typedef steam::VanillaGaussNewtonSolver SolverType;
    typedef boost::shared_ptr<SolverType> SolverBasePtr;
    SolverBasePtr solver_;
    // States
    std::deque<TrajStateVar> states_;
        adding a new frame to end based on a constant velocity extrapolation.
    */
    void slideTraj();


    /*!
       \brief Solves for the most likely set of poses and velocities given the measurements,
        constraints, and motion prior.
    */
    void optimize();

    /*!
       \brief Retrieves the estimated 4x4 poses.
        Note: poses must be sized correctly on the Python side (W x 4 x 4)
    */
    void getPoses(np::ndarray& poses);

    /*!
       \brief Retrieves the estimated velocities.
        Note: vels must be sized correctly on the Python side (W x 6)
    */
    void getVelocities(np::ndarray& vels);

    void getSigmapoints2N(np::ndarray& sigma_T);

    /*!
       \brief Interpolates for the pose of the sensor between two given times: T_ba
        which transforms points in frame a into frame b. pose must be (4 x 4) on Python side.
    */
    void getPoseBetweenTimes(np::ndarray& pose, const int64_t ta, const int64_t tb);

    void setQcInv(const np::ndarray& Qc_diag);

    /*!
       \brief Loads in the measurements to be used in the STEAM optimization.
       \param p2_list List of numpy array (N, 3)
       \param p1_list List of numpy array (N, 3)
       \param weight_list List of numpy array (N, 3, 3)
       \param t2_list List of numpy array (N,) Timestamps for points in p2_list
       \param t1_list List of numpy array (N,) Timestamps for points in p1_list
       \param t_refs List of ints Reference times at which the poses will be estimated
    */
    void setMeas(const p::object& p2_list, const p::object& p1_list, const p::object& weight_list,
        const p::object& t2_list, const p::object& t1_list, const p::object& t_refs);
    void setExtrinsicTsv(const np::ndarray& T_sv);
    void setRansacVersion(const unsigned int& version) {ransac_version = uint(version);}
    void setZeroVelPriorFlag(const bool zero_vel) {zero_vel_prior_flag_ = zero_vel;}
    void setVelPriorFlag(const bool vel_prior) {vel_prior_ = vel_prior;}
    void useRansac() {use_ransac = true;}
    void useCTSteam() {ct_steam = true;}

private:
    // Solver
    typedef steam::VanillaGaussNewtonSolver SolverType;
    typedef boost::shared_ptr<SolverType> SolverBasePtr;
    SolverBasePtr solver_;
    // States
    std::deque<TrajStateVar> states_;
    // Measurements
    std::vector<np::ndarray> p1_;           // reference
    std::vector<np::ndarray> p2_;           // frame points
    std::vector<np::ndarray> w_;            // weights
    std::vector<np::ndarray> t1_;           // reference timestamps
    std::vector<np::ndarray> t2_;           // frame points timestamps
    std::vector<int64_t> t_refs_;           // reference time for each frame
    // Constants
    double dt_ = 0.25;                      // trajectory time step
    unsigned int window_size_ = 2;          // trajectory window size
    Eigen::Matrix<double, 6, 6> Qc_inv_;    // Motion prior inverse Qc
    steam::se3::TransformEvaluator::Ptr T_sv_;
    // Configuration Flags
    bool use_ransac = false;
    unsigned int ransac_version = 0;        // 0: RIGID, 1: MC-RANSAC
    bool ct_steam = false;                  // Use timestamps for each measurement in optimization
    steam::se3::SteamTrajInterface traj;
    bool zero_vel_prior_flag_ = false;      // Apply a zero-velocity prior for dims outside SE(2)
    bool vel_prior_ = false;                // Use previously estimated velocity as another prior
    // Tracks whether the trajectory has been initialized
    bool traj_init = false;
};

// boost wrapper
BOOST_PYTHON_MODULE(SteamSolver) {
    Py_Initialize();
    np::initialize();
    p::class_<SteamSolver>("SteamSolver", p::init<const double&, const unsigned int&>())
        .def("resetTraj", &SteamSolver::resetTraj)
        .def("slideTraj", &SteamSolver::slideTraj)
        .def("setQcInv", &SteamSolver::setQcInv)
        .def("setMeas", &SteamSolver::setMeas)
        .def("setExtrinsicTsv", &SteamSolver::setExtrinsicTsv)
        .def("optimize", &SteamSolver::optimize)
        .def("getPoses", &SteamSolver::getPoses)
        .def("getVelocities", &SteamSolver::getVelocities)
        .def("useRansac", &SteamSolver::useRansac)
        .def("setRansacVersion", &SteamSolver::setRansacVersion)
        .def("setZeroVelPriorFlag", &SteamSolver::setZeroVelPriorFlag)
        .def("setVelPriorFlag", &SteamSolver::setVelPriorFlag)
        .def("useCTSteam", &SteamSolver::useCTSteam)
        .def("getPoseBetweenTimes", &SteamSolver::getPoseBetweenTimes)
        .def("getSigmapoints2N", &SteamSolver::getSigmapoints2N);
}
