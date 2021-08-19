#include <iostream>
#include "BatchSolver.hpp"
#include "P2P3ErrorEval.hpp"
#include "P2P2ErrorEval.hpp"
#include "LidarImuCalibPriorEval.hpp"
#include "mcransac.hpp"


// Set the Qc inverse matrix with the diagonal of Qc
void BatchSolver::setQcInv(const np::ndarray& Qc_diag) {
    Eigen::Matrix<double, 6, 1> temp = numpyToEigen2D(Qc_diag);
    Qc_inv_.setZero();
    Qc_inv_.diagonal() = 1.0/temp.array();
}

// Set the extrinsic matrix between vehicle and lidar
void BatchSolver::setExtrinsicLidarVehicle(const np::ndarray& T_lv) {
    Eigen::Matrix4d T_lv_eig = numpyToEigen2D(T_lv);
    lgmath::se3::Transformation T_lv_lg(T_lv_eig);
    T_lv_ = steam::se3::FixedTransformEvaluator::MakeShared(T_lv_lg);
}

// add new state variable at specified time
void BatchSolver::addNewState(double time) {
    Eigen::Matrix<double, 6, 1> zero_vel;
    zero_vel.setZero();

    // states vector
    TrajStateVar temp;
    temp.time = steam::Time(time);
    temp.pose = steam::se3::TransformStateVar::Ptr(new steam::se3::TransformStateVar(lgmath::se3::Transformation()));
    temp.velocity = steam::VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(zero_vel));
    states_.push_back(temp);

    // steam trajectory
    TrajStateVar& state = states_.back();
    steam::se3::TransformStateEvaluator::Ptr tse = steam::se3::TransformStateEvaluator::MakeShared(state.pose);
    traj_.add(state.time, tse, state.velocity);
}

// set the diagonal covariance of the extrinsic prior using a 6x1 vector
void BatchSolver::setExtrinsicPriorCov(const np::ndarray& var_diag) {
    Eigen::Matrix<double, 6, 1> temp = numpyToEigen2D(var_diag);
    extrinsic_prior_cov_.setZero();
    extrinsic_prior_cov_.diagonal() = temp.array();
}

// set the diagonal covariance of the radar keypoint evaluator using 3x1 vector
void BatchSolver::setRadarCov(const np::ndarray& var_diag) {
    Eigen::Matrix<double, 3, 1> temp = numpyToEigen2D(var_diag);
    radar_cov_.setZero();
    radar_cov_.diagonal() = temp.array();
}

// set the diagonal covariance of the lidar pose evaluator using 6x1 vector
void BatchSolver::setLidarCov(const np::ndarray& var_diag) {
    Eigen::Matrix<double, 6, 1> temp = numpyToEigen2D(var_diag);
    lidar_cov_.setZero();
    lidar_cov_.diagonal() = temp.array();
}

// input point matrices are Nx2
void BatchSolver::addFramePair(const np::ndarray& p2, const np::ndarray& p1,
    const np::ndarray& t2, const np::ndarray& t1, const int64_t& earliest_time, const int64_t& latest_time) {

    // NOTE: I think we're possibly skipping a state for the 2nd radar scan, but shouldn't matter too much
    // add new state variable if first
    if (states_.empty()) {
        time_ref_ = earliest_time;
        addNewState(0.0);
    }

    // add new state variable at latest time
    double delta_t = double(latest_time - time_ref_) / 1.0e6;
    addNewState(delta_t);
    std::cout << "Added new state at: " << delta_t << " seconds" << std::endl;

    // ransac
    std::vector<int> inliers;
    if (use_ransac) {
        srand(time_ref_ / 1.0e6);  // fix random seed for repeatability
        Eigen::VectorXd motion_vec = Eigen::VectorXd::Zero(6);
        Eigen::MatrixXd T;

        Ransac ransac(p1, p2);
        ransac.computeModel();
        ransac.getTransform(T);
        ransac.getInliers(T, inliers);

    } else {
        for (uint j = 0; j < p1.shape(0); ++j) {
            inliers.push_back(j);
        }
    }

    // choose between L2 and GemanMcClure
    steam::LossFunctionBase::Ptr sharedLossFunc;
    if (use_radar_robust_cost_)
        sharedLossFunc = steam::GemanMcClureLossFunc::Ptr(new steam::GemanMcClureLossFunc(1.0));
    else
        sharedLossFunc = steam::L2LossFunc::Ptr(new steam::L2LossFunc());

    // loop through each match pair
    for (uint k = 0; k < inliers.size(); ++k) {
        uint j = inliers[k];    // index of inlier

        // get measurement
        Eigen::Vector4d read;
        read << double(p::extract<float>(p2[j][0])), double(p::extract<float>(p2[j][1])), 0.0, 1.0;

        Eigen::Vector4d ref;
        ref << double(p::extract<float>(p1[j][0])), double(p::extract<float>(p1[j][1])), 0.0, 1.0;

        // get relative pose expression
        int64_t ta_ = int64_t(p::extract<int64_t>(t1[j])) - time_ref_;
        int64_t tb_ = int64_t(p::extract<int64_t>(t2[j])) - time_ref_;
        double ta = double(ta_) / 1.0e6;
        double tb = double(tb_) / 1.0e6;
        // steam::se3::TransformStateEvaluator::Ptr Tsv = steam::se3::TransformStateEvaluator::MakeShared(T_s_v_);

        steam::se3::TransformStateEvaluator::Ptr Tfl = steam::se3::TransformStateEvaluator::MakeShared(T_fl_);
        steam::se3::TransformEvaluator::Ptr Trl = steam::se3::compose(T_rf_, Tfl);
        steam::se3::TransformEvaluator::Ptr Trv = steam::se3::compose(Trl, T_lv_);

        steam::se3::TransformEvaluator::ConstPtr Ta0 = traj_.getInterpPoseEval(steam::Time(ta));
        steam::se3::TransformEvaluator::ConstPtr Tb0 = traj_.getInterpPoseEval(steam::Time(tb));
        steam::se3::TransformEvaluator::Ptr T_eval_ptr = steam::se3::composeInverse(
            steam::se3::compose(Trv, Tb0),
            steam::se3::compose(Trv, Ta0));  // Tba = Tb0 * inv(Ta0)

        // add cost
        if (radar_2d_error_) {
            Eigen::Matrix2d R = radar_cov_.block<2, 2>(0, 0);
            steam::BaseNoiseModel<2>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<2>(R, steam::COVARIANCE));
            steam::P2P2ErrorEval::Ptr error(new steam::P2P2ErrorEval(ref, read, T_eval_ptr));
            steam::WeightedLeastSqCostTerm<2, 6>::Ptr cost(
                new steam::WeightedLeastSqCostTerm<2, 6>(error, sharedNoiseModel, sharedLossFunc));
            radar_cost_terms_->add(cost);
        }
        else {
            steam::BaseNoiseModel<3>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<3>(radar_cov_, steam::COVARIANCE));
            steam::P2P3ErrorEval::Ptr error(new steam::P2P3ErrorEval(ref, read, T_eval_ptr));
            steam::WeightedLeastSqCostTerm<3, 6>::Ptr cost(
                new steam::WeightedLeastSqCostTerm<3, 6>(error, sharedNoiseModel, sharedLossFunc));
            radar_cost_terms_->add(cost);
        }
    }
}

void BatchSolver::addLidarPoses(const p::object& T_il_list, const p::object& times_list) {
    std::vector<np::ndarray> T_il_vec = toStdVector<np::ndarray>(T_il_list);
    std::vector<int64_t> times_vec = toStdVector<int64_t>(times_list);

    steam::BaseNoiseModel<6>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<6>(lidar_cov_));
    steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());

    for (int i = 0; i < T_il_vec.size(); ++i) {
        // skip if before first state time
        if (times_vec[i] <= getFirstStateTime())
            continue;

        // break if after last state time
        if (times_vec[i] >= getLastStateTime())
            break;

        // add pose measurement
        Eigen::Matrix<double,4,4> T_il_eig = numpyToEigen2D(T_il_vec[i]);
        lgmath::se3::Transformation T_il_meas(T_il_eig);
        int64_t time_nano = times_vec[i] - time_ref_*1e3;
        double time_sec = double(time_nano) / 1.0e9;
        std::cout << "Adding lidar pose measurement (" << i << ") at: " << time_sec << " seconds" << std::endl;
        steam::se3::TransformEvaluator::ConstPtr T_vi = traj_.getInterpPoseEval(steam::Time(time_sec));
//        steam::TransformErrorEval::Ptr posefunc(new steam::TransformErrorEval(T_il_meas.inverse(),
//            steam::se3::ComposeTransformEvaluator::MakeShared(T_lv_, T_vi)));
        steam::TransformErrorEval::Ptr posefunc(new steam::TransformErrorEval(T_il_meas.inverse(), T_vi));
        steam::WeightedLeastSqCostTerm<6,6>::Ptr cost(new steam::WeightedLeastSqCostTerm<6,6>(
            posefunc, sharedNoiseModel, sharedLossFunc));
        lidar_cost_terms_->add(cost);
    }
}

void BatchSolver::addLidarPosesRel(const p::object& T_il_list, const p::object& times_list) {
    std::vector<np::ndarray> T_il_vec = toStdVector<np::ndarray>(T_il_list);
    std::vector<int64_t> times_vec = toStdVector<int64_t>(times_list);

    Eigen::Matrix<double,6,6> cov_eig = 1e-2*Eigen::Matrix<double,6,6>::Identity();
    cov_eig(3, 3) = 1e-4;
    cov_eig(4, 4) = 1e-4;
    cov_eig(5, 5) = 1e-4;   // TODO: tune through config
    steam::BaseNoiseModel<6>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<6>(cov_eig));
    steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());

    for (int i = 0; i < T_il_vec.size() - 1; ++i) {
        // skip if before first state time
        if (times_vec[i] <= getFirstStateTime())
            continue;

        // break if after last state time
        if (times_vec[i + 1] >= getLastStateTime())
            break;

        // add relative pose measurement
        Eigen::Matrix<double,4,4> temp = numpyToEigen2D(T_il_vec[i]);
        lgmath::se3::Transformation T_i_l1_meas(temp);
        int64_t time1_nano = times_vec[i] - time_ref_*1e3;
        double time1_sec = double(time1_nano) / 1.0e9;

        temp = numpyToEigen2D(T_il_vec[i+1]);
        lgmath::se3::Transformation T_i_l2_meas(temp);
        int64_t time2_nano = times_vec[i+1] - time_ref_*1e3;
        double time2_sec = double(time2_nano) / 1.0e9;

        std::cout << "Adding lidar relative pose measurement (" << i << ", " << i+1 << ") at: " << time1_sec << " and "
            << time2_sec <<" seconds" << std::endl;

        steam::se3::TransformEvaluator::ConstPtr T_v1_i = traj_.getInterpPoseEval(steam::Time(time1_sec));
        steam::se3::TransformEvaluator::ConstPtr T_v2_i = traj_.getInterpPoseEval(steam::Time(time2_sec));

//        steam::se3::TransformEvaluator::Ptr T_l2_l1 = steam::se3::composeInverse(
//            steam::se3::compose(T_lv_, T_v2_i),
//            steam::se3::compose(T_lv_, T_v1_i));  // Tba = Tb0 * inv(Ta0)
        steam::se3::TransformEvaluator::Ptr T_l2_l1 = steam::se3::composeInverse(
            T_v2_i,
            T_v1_i);  // Tba = Tb0 * inv(Ta0)

        steam::TransformErrorEval::Ptr posefunc(new steam::TransformErrorEval(T_i_l2_meas.inverse()*T_i_l1_meas,
            T_l2_l1));
        steam::WeightedLeastSqCostTerm<6,6>::Ptr cost(new steam::WeightedLeastSqCostTerm<6,6>(
            posefunc, sharedNoiseModel, sharedLossFunc));
        lidar_cost_terms_->add(cost);
    }
}

// Run optimization
void BatchSolver::optimize() {

    // lock first pose
    if (lock_first_pose_) {
        states_[0].pose->setLock(true);
//        T_fl_->setLock(true);  // not solving for extrinsic if no lidar poses
    }
    else {
        states_[0].pose->setLock(false);
    }

    // additional cost terms
    steam::ParallelizedCostTermCollection::Ptr costs(new steam::ParallelizedCostTermCollection());

    // prior on extrinsic
//    steam::L2LossFunc::Ptr sharedLossFuncL2(new steam::L2LossFunc());
//    Eigen::Matrix3d U = Eigen::Matrix3d::Identity();
//    U(0, 0) = 0.01*0.01;    // variance of z-offset
//    U(1, 1) = 0.001*0.001;  // variance of roll-offset
//    U(2, 2) = 0.001*0.001;  // variance of elevation-offset
//    steam::BaseNoiseModel<3>::Ptr sharedNoiseModelU(new steam::StaticNoiseModel<3>(U, steam::COVARIANCE));
//    steam::se3::TransformStateEvaluator::Ptr Tfl = steam::se3::TransformStateEvaluator::MakeShared(T_fl_);
//    steam::LidarImuCalibPriorEval::Ptr error(new steam::LidarImuCalibPriorEval(z_offset_, Tfl));
//    steam::WeightedLeastSqCostTerm<3, 6>::Ptr costU(
//        new steam::WeightedLeastSqCostTerm<3, 6>(error, sharedNoiseModelU, sharedLossFuncL2));
//    costs->add(costU);

    steam::se3::TransformStateEvaluator::Ptr Tfl = steam::se3::TransformStateEvaluator::MakeShared(T_fl_);
    if (Tfl->isActive()) {
        steam::L2LossFunc::Ptr sharedLossFuncL2(new steam::L2LossFunc());
//        Eigen::Matrix<double,6,6> R = Eigen::Matrix<double,6,6>::Zero();
//        Eigen::Array<double, 1, 6> R_diag;
//        R_diag << 0.0001, 0.0001, 0.0001, 8e-5, 8e-5, 2.46741264;
//        R.diagonal() = R_diag;
        steam::BaseNoiseModel<6>::Ptr sharedNoiseModel(
            new steam::StaticNoiseModel<6>(extrinsic_prior_cov_, steam::COVARIANCE));

        Eigen::Matrix4d Tfl_eig = Eigen::Matrix4d::Identity();
        Tfl_eig(2, 3) = z_offset_;
        Tfl_eig(1, 1) = -1.0;
        Tfl_eig(2, 2) = -1.0;
        lgmath::se3::Transformation Tfl_meas(Tfl_eig);
        steam::TransformErrorEval::Ptr posefunc(new steam::TransformErrorEval(Tfl_meas, Tfl));
        steam::WeightedLeastSqCostTerm<6,6>::Ptr cost(new steam::WeightedLeastSqCostTerm<6,6>(
            posefunc, sharedNoiseModel, sharedLossFuncL2));
        costs->add(cost);
    }

    // WNOA
    std::cout << "Getting WNOA prior terms..." << std::endl;
    traj_.appendPriorCostTerms(costs);

    steam::OptimizationProblem problem;

    // Add state variables
    std::cout << "Adding state variables..." << std::endl;
    for (uint i = 0; i < states_.size(); ++i) {
        const TrajStateVar& state = states_.at(i);
        problem.addStateVariable(state.pose);
        problem.addStateVariable(state.velocity);
    }
    problem.addStateVariable(T_fl_);   // extrinsic

    std::cout << "Adding cost terms..." << std::endl;
    problem.addCostTerm(costs);
    if (use_radar_)
        problem.addCostTerm(radar_cost_terms_);
    if (use_lidar_)
        problem.addCostTerm(lidar_cost_terms_);
    SolverType::Params params;
    params.verbose = true;
    solver_ = SolverBasePtr(new SolverType(&problem, params));

    std::cout << "Optimizing..." << std::endl;
    solver_->optimize();
    std::cout << "Complete." << std::endl;

    std::cout << "T_rf:" << std::endl << T_rf_->evaluate().matrix() << std::endl << std::endl;
    std::cout << "T_fl:" << std::endl << T_fl_->getValue().matrix() << std::endl << std::endl;
    std::cout << "T_lv:" << std::endl << T_lv_->evaluate().matrix() << std::endl << std::endl;

    std::cout << T_rf_->evaluate().matrix()*T_fl_->getValue().matrix() << std::endl << std::endl;
}

void BatchSolver::getPoses(np::ndarray& poses) {
    for (uint i = 0; i < states_.size(); ++i) {
//        Eigen::Matrix<double, 4, 4> Tsi =
//            T_s_v_->getValue().matrix()*states_[i].pose->getValue().matrix()*T_s_v_->getValue().inverse().matrix();
        Eigen::Matrix<double, 4, 4> Tvi = states_[i].pose->getValue().matrix();
        for (uint r = 0; r < 3; ++r) {
            for (uint c = 0; c < 4; ++c) {
//                poses[i][r][c] = float(Tsi(r, c));
                poses[i][r][c] = float(Tvi(r, c));
            }
        }
    }
}

void BatchSolver::getPath(np::ndarray& path, np::ndarray& times) {
    for (uint i = 0; i < states_.size(); ++i) {
        Eigen::Matrix<double, 3, 1> r_vi_in_i = states_[i].pose->getValue().r_ba_ina();
//        Eigen::Matrix<double, 3, 1> r_vi_in_i =
//            (states_[i].pose->getValue()*states_[0].pose->getValue().inverse()).r_ba_ina();
        for (uint r = 0; r < 3; ++r) {
            path[i][r] = float(r_vi_in_i(r));
        }
        times[i] = float(states_[i].time.seconds());
    }
}

void BatchSolver::getRadarLidarExtrinsic(np::ndarray& T_rl) {
    Eigen::Matrix4d T_rl_eig = T_rf_->evaluate().matrix()*T_fl_->getValue().matrix();
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            T_rl[r][c] = float(T_rl_eig(r, c));
        }
    }
}

void BatchSolver::getVelocities(np::ndarray& vels) {
    for (uint i = 0; i < states_.size(); ++i) {
        Eigen::Matrix<double, 6, 1> vel = states_[i].velocity->getValue();
        for (uint r = 0; r < 6; ++r) {
            vels[i][r] = float(vel(r));
        }
    }
}

void BatchSolver::undistortPointcloud(np::ndarray& points, const np::ndarray& times,
        const int64_t& ref_time, const np::ndarray& Trl_np, const int& mode) {
    // get vehicle pose at ref_time
    double t_ref_sec = double(ref_time - time_ref_) / 1.0e6;
    steam::se3::TransformEvaluator::ConstPtr T_ref_i = traj_.getInterpPoseEval(steam::Time(t_ref_sec));

    // get extrinsic
    steam::se3::TransformEvaluator::Ptr Tsv;
    if (mode == 0) {    // radar
        steam::se3::TransformStateEvaluator::Ptr Tfl = steam::se3::TransformStateEvaluator::MakeShared(T_fl_);
        steam::se3::TransformEvaluator::Ptr Trl = steam::se3::compose(T_rf_, Tfl);
        Tsv = steam::se3::compose(Trl, T_lv_);
    }
    else if (mode == 1) {   // radar but with given extrinsic
        Eigen::Matrix4d Trl_eig = numpyToEigen2D(Trl_np);
        steam::se3::TransformEvaluator::Ptr Trl =
            steam::se3::FixedTransformEvaluator::MakeShared(lgmath::se3::Transformation(Trl_eig));
        Tsv = steam::se3::compose(Trl, T_lv_);
    }
    else {  // lidar
        Tsv = T_lv_;
    }

    // loop through every point
    for (int i = 0; i < points.shape(0); ++i) {
        // interpolate for pose
        int64_t ta_ = int64_t(p::extract<int64_t>(times[i])) - time_ref_;
        double ta = double(ta_) / 1.0e6;
        steam::se3::TransformEvaluator::ConstPtr T_a_i = traj_.getInterpPoseEval(steam::Time(ta));

        // transform to ref
        Eigen::Vector4d point;
        if (mode == 0 || mode == 1) { // radar
            point << double(p::extract<float>(points[i][0])), double(p::extract<float>(points[i][1])), 0.0, 1.0;
        }
        else {  // lidar
            point << double(p::extract<float>(points[i][0])), double(p::extract<float>(points[i][1])),
                     double(p::extract<float>(points[i][2])), 1.0;
        }
        Eigen::Vector4d point_ref = T_ref_i->evaluate().matrix() * T_a_i->evaluate().inverse().matrix()
            * Tsv->evaluate().inverse().matrix() * point;

        // write out
        for (uint r = 0; r < points.shape(1); ++r) {
            points[i][r] = float(point_ref(r));
        }
    }
}
