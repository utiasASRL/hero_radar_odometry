#pragma once
#include <vector>
#include <deque>
#include "SteamPyHelper.hpp"

class SteamSolver {
public:
    SteamSolver(const double& dt, const unsigned int& window_size) :
        dt_(dt), window_size_(window_size) {
        Eigen::Array<double, 1, 6> Qc_diag;
        Qc_diag << 0.3678912639416186958207788393338,
                   0.043068034591947058908889545136844,
                   0.1307444996557916849777569723301,
                   0.0073124100132336252236275875304727,
                   0.0076438703775169331705585662461999,
                   0.0021394075786459413462958778495704;
        Qc_inv_.setZero();
        Qc_inv_.diagonal() = 1.0/Qc_diag;

        // Initialize extrinsic transform to identity
        T_sv_ = steam::se3::FixedTransformEvaluator::MakeShared(lgmath::se3::Transformation());

        // Initialize trajectory
        resetTraj();
    }
    // initialization
    void resetTraj();
    void slideTraj();
    void setQcInv(const np::ndarray& Qc_diag);
    void setMeas(const p::object& p2_list, const p::object& p1_list, const p::object& weight_list,
        const p::object& t2_list, const p::object& t1_list, const p::object& t_refs);
    void setExtrinsicTsv(const np::ndarray& T_sv);
    // solve
    void optimize();
    // output
    void getPoses(np::ndarray& poses);
    void getVelocities(np::ndarray& vels);
    void getSigmapoints2N(np::ndarray& sigma_T);
    void useRansac() {use_ransac = true;}
    void setRansacVersion(const unsigned int& version) {ransac_version = uint(version);}
    void useCTSteam() {ct_steam = true;}
    void getPoseBetweenTimes(np::ndarray& pose, const int64_t ta, const int64_t tb);

private:
    // Solver
    typedef steam::VanillaGaussNewtonSolver SolverType;
    typedef boost::shared_ptr<SolverType> SolverBasePtr;
    SolverBasePtr solver_;
    // States
    std::deque<TrajStateVar> states_;
    // Measurements
    std::vector<np::ndarray> p1_;  // reference
    std::vector<np::ndarray> p2_;  // frame points
    std::vector<np::ndarray> w_;   // weights
    std::vector<np::ndarray> t1_;  // reference timestamps
    std::vector<np::ndarray> t2_;  // frame points timestamps
    std::vector<int64_t> t_refs_;  // time for each frame
    // Constants
    double dt_ = 0.25;  // trajectory time step
    unsigned int window_size_ = 2;  // trajectory window size
    Eigen::Matrix<double, 6, 6> Qc_inv_;  // Motion prior inverse Qc
    steam::se3::TransformEvaluator::Ptr T_sv_;
    // RANSAC
    bool use_ransac = false;
    unsigned int ransac_version = 0;
    bool ct_steam = false;
    steam::se3::SteamTrajInterface traj;
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
        .def("useCTSteam", &SteamSolver::useCTSteam)
        .def("getPoseBetweenTimes", &SteamSolver::getPoseBetweenTimes)
        .def("getSigmapoints2N", &SteamSolver::getSigmapoints2N);
}
