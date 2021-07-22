#pragma once
#include <vector>
#include <deque>
#include "SteamPyHelper.hpp"

class BatchSolver {
public:
    BatchSolver() {
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
//        T_sv_ = steam::se3::FixedTransformEvaluator::MakeShared(lgmath::se3::Transformation());
//        lgmath::se3::Transformation T_v_s(T_v_s_eig);
        T_s_v_ = steam::se3::TransformStateVar::Ptr(new steam::se3::TransformStateVar(lgmath::se3::Transformation()));
//        steam::ParallelizedCostTermCollection::Ptr costTerms(new steam::ParallelizedCostTermCollection());
        cost_terms_ = steam::ParallelizedCostTermCollection::Ptr(new steam::ParallelizedCostTermCollection());
    }
    // initialization
    void setQcInv(const np::ndarray& Qc_diag);
//    void setExtrinsicTsv(const np::ndarray& T_sv); // states_[k].pose->setValue(lgmath::se3::Transformation(xi)*T_km1_i);

    // solve
    void addFramePair(const np::ndarray& p2, const np::ndarray& p1,
                      const np::ndarray& t2, const np::ndarray& t1,
                      const int64_t& earliest_time, const int64_t& latest_time);
    void optimize();

    // output
    void getPoses(np::ndarray& poses);
    void getVelocities(np::ndarray& vels);
    void useRansac() {use_ransac = true;}

private:
    // Solver
    typedef steam::VanillaGaussNewtonSolver SolverType;
    typedef boost::shared_ptr<SolverType> SolverBasePtr;
    SolverBasePtr solver_;
    steam::se3::SteamTrajInterface traj_;
    steam::ParallelizedCostTermCollection::Ptr cost_terms_;
    void addNewState(double time);

    // States
    std::deque<TrajStateVar> states_;
    steam::se3::TransformStateVar::Ptr T_s_v_;

    // Constants
    Eigen::Matrix<double, 6, 6> Qc_inv_;  // Motion prior inverse Qc
    int64_t time_ref_ = 0;

    // RANSAC
    bool use_ransac = true;
};

// boost wrapper
BOOST_PYTHON_MODULE(BatchSolver) {
    Py_Initialize();
    np::initialize();
    p::class_<BatchSolver>("BatchSolver", p::init<>())
        .def("setQcInv", &BatchSolver::setQcInv)
        .def("addFramePair", &BatchSolver::addFramePair)
        .def("optimize", &BatchSolver::optimize)
        .def("getPoses", &BatchSolver::getPoses)
        .def("getVelocities", &BatchSolver::getVelocities)
        .def("useRansac", &BatchSolver::useRansac);
}
