#pragma once
#include <vector>
#include <deque>
#include "SteamPyHelper.hpp"

class BatchSolver {
public:
    BatchSolver() {
        Eigen::Array<double, 1, 6> Qc_diag;
//        Qc_diag << 0.3678912639416186958207788393338,
//                   0.043068034591947058908889545136844,
//                   0.1307444996557916849777569723301,
//                   0.0073124100132336252236275875304727,
//                   0.0076438703775169331705585662461999,
//                   0.0021394075786459413462958778495704;
        Qc_diag << 1.0, 0.1, 0.1, 0.1, 0.1, 1.0;
        Qc_inv_.setZero();
        Qc_inv_.diagonal() = 1.0/Qc_diag;

        // Initialize extrinsic transforms
        // T_lv:
        // 0.686 -0.727     -0     -0
        // 0.727  0.686      0      0
        //     0      0      1  -0.21
        //     0      0      0      1
        Eigen::Matrix<double,4,4> T_fl_eig;
        T_fl_eig << 0.686,  0.727, 0, 0,
                    -0.727, 0.686, 0, 0,
                    0,      0,     1, 0,
                    0,      0,     0, 1;
//        T_fl_ = steam::se3::TransformStateVar::Ptr(new steam::se3::TransformStateVar(lgmath::se3::Transformation(T_fl_eig)));
        T_fl_ = steam::se3::TransformStateVar::Ptr(new steam::se3::TransformStateVar(lgmath::se3::Transformation()));


        T_lv_ = steam::se3::FixedTransformEvaluator::MakeShared(lgmath::se3::Transformation());

        Eigen::Matrix<double,4,4> T_rf_eig;
        T_rf_eig << 1,  0,  0, 0,
                    0, -1,  0, 0,
                    0,  0, -1, 0,
                    0,  0,  0, 1;
        lgmath::se3::Transformation T_rf(T_rf_eig);
        T_rf_ = steam::se3::FixedTransformEvaluator::MakeShared(T_rf);

//        steam::ParallelizedCostTermCollection::Ptr costTerms(new steam::ParallelizedCostTermCollection());
        radar_cost_terms_ = steam::ParallelizedCostTermCollection::Ptr(new steam::ParallelizedCostTermCollection());
        lidar_cost_terms_ = steam::ParallelizedCostTermCollection::Ptr(new steam::ParallelizedCostTermCollection());
    }
    // initialization
    void setQcInv(const np::ndarray& Qc_diag);
    void setExtrinsicLidarVehicle(const np::ndarray& T_lv);
    void setLockExtrinsicState(const bool& flag) {
        T_fl_->setLock(flag);
    }
    void setRadarFlag(const bool& flag) {
        use_radar_ = flag;
    }
    void setLidarFlag(const bool& flag) {
        use_lidar_ = flag;
    }
    void setRadar2DFlag(const bool& flag) {
        radar_2d_error_ = flag;
    }
    void setFirstPoseLock(const bool& flag) {
        lock_first_pose_ = flag;
    }

    // solve
    void addFramePair(const np::ndarray& p2, const np::ndarray& p1,
                      const np::ndarray& t2, const np::ndarray& t1,
                      const int64_t& earliest_time, const int64_t& latest_time);
    void addLidarPoses(const p::object& T_il_list, const p::object& times_list);
    void addLidarPosesRel(const p::object& T_il_list, const p::object& times_list);
    void optimize();

    // output
    int getTrajLength() {
        return states_.size();
    }
    void getPoses(np::ndarray& poses);
    void getPath(np::ndarray& path, np::ndarray& times);
    void getVelocities(np::ndarray& vels);
    void useRansac() {use_ransac = true;}
    int64_t getFirstStateTime() {
        return states_.front().time.nanosecs() + time_ref_*1e3;
    }
    int64_t getLastStateTime() {
        return states_.back().time.nanosecs() + time_ref_*1e3;
    }

private:
    // Solver
    typedef steam::VanillaGaussNewtonSolver SolverType;
    typedef boost::shared_ptr<SolverType> SolverBasePtr;
    SolverBasePtr solver_;
    steam::se3::SteamTrajInterface traj_;
    steam::ParallelizedCostTermCollection::Ptr radar_cost_terms_;
    steam::ParallelizedCostTermCollection::Ptr lidar_cost_terms_;
    void addNewState(double time);

    // States
    std::deque<TrajStateVar> states_;
    steam::se3::TransformStateVar::Ptr T_fl_;

    // Extrinsics
    // frames: v (vehicle), l (lidar), f (flipped radar, i.e., z-up +), r (radar frame)
    // T_rv = T_rf x T_fl x T_lv
    // T_lv and T_rf are known constants
//    steam::se3::TransformEvaluator::Ptr T_sv_;
    steam::se3::TransformEvaluator::Ptr T_lv_;
    steam::se3::TransformEvaluator::Ptr T_rf_;

    // Constants
    Eigen::Matrix<double, 6, 6> Qc_inv_;  // Motion prior inverse Qc
    int64_t time_ref_ = 0;
    double z_offset_ = -0.21;

    // flags
    bool use_ransac = true;
    bool use_radar_ = true;
    bool use_lidar_ = true;
    bool radar_2d_error_ = true;
    bool lock_first_pose_ = true;
};

// boost wrapper
BOOST_PYTHON_MODULE(BatchSolver) {
    Py_Initialize();
    np::initialize();
    p::class_<BatchSolver>("BatchSolver", p::init<>())
        .def("setQcInv", &BatchSolver::setQcInv)
        .def("setExtrinsicLidarVehicle", &BatchSolver::setExtrinsicLidarVehicle)
        .def("setLockExtrinsicState", &BatchSolver::setLockExtrinsicState)
        .def("setRadarFlag", &BatchSolver::setRadarFlag)
        .def("setLidarFlag", &BatchSolver::setLidarFlag)
        .def("setRadar2DFlag", &BatchSolver::setRadar2DFlag)
        .def("setFirstPoseLock", &BatchSolver::setFirstPoseLock)
        .def("addFramePair", &BatchSolver::addFramePair)
        .def("addLidarPoses", &BatchSolver::addLidarPoses)
        .def("addLidarPosesRel", &BatchSolver::addLidarPosesRel)
        .def("optimize", &BatchSolver::optimize)
        .def("getPoses", &BatchSolver::getPoses)
        .def("getPath", &BatchSolver::getPath)
        .def("getVelocities", &BatchSolver::getVelocities)
        .def("useRansac", &BatchSolver::useRansac)
        .def("getTrajLength", &BatchSolver::getTrajLength)
        .def("getFirstStateTime", &BatchSolver::getFirstStateTime)
        .def("getLastStateTime", &BatchSolver::getLastStateTime);
}
