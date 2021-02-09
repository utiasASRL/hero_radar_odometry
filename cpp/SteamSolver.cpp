#include "SteamSolver.hpp"
#include "P2P3ErrorEval.hpp"
#include "SE2VelPriorEval.hpp"

// Reset trajectory to identity poses and zero velocities
void SteamSolver::resetTraj() {
    Eigen::Matrix<double, 4, 4> eig_identity = Eigen::Matrix<double, 4, 4>::Identity();
    lgmath::se3::Transformation T_identity(eig_identity);
    Eigen::Matrix<double, 6, 1> zero_vel;
    zero_vel.setZero();
    states_.clear();
    for (uint k = 0; k < window_size_; ++k) {
        TrajStateVar temp;
        temp.time = steam::Time(k * dt_);
        temp.pose = steam::se3::TransformStateVar::Ptr(new steam::se3::TransformStateVar(T_identity));
        temp.velocity = steam::VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(zero_vel));
        states_.push_back(temp);
    }
}

// Slide window and initialize newest frame with constant velocity
void SteamSolver::slideTraj() {
    // drop first frame
    states_.pop_front();

    // set first frame to identity
    lgmath::se3::Transformation T_i0 = states_[0].pose->getValue().inverse();
    for (uint k = 0; k < states_.size(); ++k){
        lgmath::se3::Transformation T_ki = states_[k].pose->getValue();
        states_[k].pose->setValue(T_ki*T_i0);
    }

    // add new frame to end
    lgmath::se3::Transformation T_km1_i = states_.back().pose->getValue();
    Eigen::Matrix<double, 6, 1> xi = dt_ * states_.back().velocity->getValue();

    TrajStateVar temp;
    temp.time = states_.back().time + steam::Time(dt_);
    temp.pose = steam::se3::TransformStateVar::Ptr(new steam::se3::TransformStateVar(
        lgmath::se3::Transformation(xi)*T_km1_i));
    temp.velocity = steam::VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(
        states_.back().velocity->getValue()));
    states_.push_back(temp);
}

// Set the Qc inverse matrix with the diagonal of Qc
void SteamSolver::setQcInv(const np::ndarray& Qc_diag) {
    Eigen::Matrix<double, 6, 1> temp = numpyToEigen2D(Qc_diag);
    Qc_inv_.setZero();
    Qc_inv_.diagonal() = 1.0/temp.array();
}

// Set measurements
void SteamSolver::setMeas(const p::object& p2_list,
    const p::object& p1_list, const p::object& weight_list) {
    p2_ = toStdVector<np::ndarray>(p2_list);
    p1_ = toStdVector<np::ndarray>(p1_list);
    w_ = toStdVector<np::ndarray>(weight_list);
}

// Run optimization
void SteamSolver::optimize() {
    // Motion prior
    steam::se3::SteamTrajInterface traj(Qc_inv_);
    for (uint i = 0; i < states_.size(); ++i) {
        TrajStateVar& state = states_.at(i);
        steam::se3::TransformStateEvaluator::Ptr temp = steam::se3::TransformStateEvaluator::MakeShared(state.pose);
        traj.add(state.time, temp, state.velocity);
        if (i == 0)  // lock first pose
            state.pose->setLock(true);
    }  // end i

    // Cost Terms
    steam::ParallelizedCostTermCollection::Ptr costTerms(new steam::ParallelizedCostTermCollection());
    traj.appendPriorCostTerms(costTerms);

    steam::L2LossFunc::Ptr sharedLossFuncL2(new steam::L2LossFunc());
    steam::GemanMcClureLossFunc::Ptr sharedLossFuncGM(new steam::GemanMcClureLossFunc(1.0));

    // loop through every frame
    int src_id = window_size_ - 1;
    for (uint i = 0; i < window_size_ - 1; ++i) {
        int tgt_id = i;
        steam::se3::TransformEvaluator::Ptr Ta0 = steam::se3::TransformStateEvaluator::MakeShared(states_[src_id].pose);
        steam::se3::TransformEvaluator::Ptr Tb0 = steam::se3::TransformStateEvaluator::MakeShared(states_[tgt_id].pose);
        steam::se3::TransformEvaluator::Ptr T_eval_ptr = steam::se3::composeInverse(Tb0, Ta0);  // Tba = Tb0 * inv(Ta0)
        uint num_meas = p2_[i].shape(0);
        for (uint j = 0; j < num_meas; ++j) {
            Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
            for (uint r = 0; r < 3; ++r) {
                for (uint c = 0; c < 3; ++c) {
                    R(r, c) = p::extract<float>(w_[i][j][r][c]);
                }
            }
            steam::BaseNoiseModel<3>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<3>(R, steam::INFORMATION));
            // get measurement
            Eigen::Vector4d read;
            read << double(p::extract<float>(p2_[i][j][0])), double(p::extract<float>(p2_[i][j][1])),
                  double(p::extract<float>(p2_[i][j][2])), 1.0;

            Eigen::Vector4d ref;
            ref << double(p::extract<float>(p1_[i][j][0])), double(p::extract<float>(p1_[i][j][1])),
                 double(p::extract<float>(p1_[i][j][2])), 1.0;

            steam::P2P3ErrorEval::Ptr error(new steam::P2P3ErrorEval(ref, read, T_eval_ptr));
            steam::WeightedLeastSqCostTerm<3, 6>::Ptr cost(
                new steam::WeightedLeastSqCostTerm<3, 6>(error, sharedNoiseModel, sharedLossFuncGM));
            costTerms->add(cost);
        }  // end j
    }  // end i

    // SE(2) velocity priors
    // TODO(david): make vel_prior_noise a parameter
    if (zero_vel_prior_flag_) {
        Eigen::Matrix<double, 3, 3> vel_prior_noise = 1e-3 * Eigen::Matrix<double, 3, 3>::Identity();
        steam::BaseNoiseModel<3>::Ptr vel_prior_noise_model(new steam::StaticNoiseModel<3>(vel_prior_noise));
        for (uint i = 0; i < states_.size(); ++i) {
            steam::SE2VelPriorEval::Ptr error(new steam::SE2VelPriorEval(states_[i].velocity));
            steam::WeightedLeastSqCostTerm<3, 6>::Ptr cost(
                new steam::WeightedLeastSqCostTerm<3, 6>(error, vel_prior_noise_model, sharedLossFuncL2));
            costTerms->add(cost);
        }  // end i
    }

    // Initialize problem
    steam::OptimizationProblem problem;
    // Add state variables
    for (uint i = 0; i < states_.size(); ++i) {
        const TrajStateVar& state = states_.at(i);
        problem.addStateVariable(state.pose);
        problem.addStateVariable(state.velocity);
    }
    // Add cost terms
    problem.addCostTerm(costTerms);
    // Solver parameters
    // TODO(david): Make this a parameter
    // typedef steam::DoglegGaussNewtonSolver SolverType;
    // typedef steam::LevMarqGaussNewtonSolver SolverType;
    // typedef steam::VanillaGaussNewtonSolver SolverType;
    SolverType::Params params;
    params.verbose = false;  // TODO(david): make this a parameter
    // Make solver
    solver_ = SolverBasePtr(new SolverType(&problem, params));
    // Optimize
    solver_->optimize();
}

void SteamSolver::getPoses(np::ndarray& poses) {
    for (uint i = 0; i < states_.size(); ++i) {
        // get position
        Eigen::Matrix<double, 4, 4> Tvi = states_[i].pose->getValue().matrix();
        // set output
        for (uint r = 0; r < 3; ++r) {
            for (uint c = 0; c < 4; ++c) {
                poses[i][r][c] = float(Tvi(r, c));
            }
        }
    }
}

void SteamSolver::getVelocities(np::ndarray& vels) {
    for (uint i = 0; i < states_.size(); ++i) {
        // get position
        Eigen::Matrix<double, 6, 1> vel = states_[i].velocity->getValue();
        // set output
        for (uint r = 0; r < 6; ++r) {
            vels[i][r] = float(vel(r));
        }
    }
}

void SteamSolver::getSigmapoints2NP1(np::ndarray& sigma_T) {
    // query covariance at once
    std::vector<steam::StateKey> keys;
    keys.reserve(window_size_ - 1);
    for (unsigned int i = 1; i < states_.size(); i++) {
        // skip i = 0 since it's always locked
        const TrajStateVar& state = states_.at(i);
        keys.push_back(state.pose->getKey());
    }
    steam::BlockMatrix cov_blocks = solver_->queryCovarianceBlock(keys);

    // useful constants
    int n = 6;  // pose is 6D
    double alpha = sqrt(double(n));

    // loop through every frame (skipping first since it's locked)
    for (unsigned int i = 1; i < window_size_; i++) {
        // mean pose
        const TrajStateVar& state = states_.at(i);
        Eigen::Matrix4d T_i0_eigen = state.pose->getValue().matrix();

        // get cov and LLT decomposition
        Eigen::Matrix<double, 6, 6> cov = cov_blocks.at(i - 1, i - 1);
        Eigen::LLT<Eigen::MatrixXd> lltcov(cov);
        Eigen::MatrixXd L = lltcov.matrixL();

        // sigmapoints
        for (int a = 0; a < n; ++a) {
            // delta for pose
            Eigen::Matrix4d T_sp = lgmath::se3::vec2tran(L.col(a).head<6>()*alpha);
            Eigen::Matrix4d T_sp_inv = lgmath::se3::vec2tran(-L.col(a).head<6>()*alpha);

            // positive/negative sigmapoints
            T_sp = T_sp*T_i0_eigen;
            T_sp_inv = T_sp_inv*T_i0_eigen;

            // set output
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    sigma_T[i-1][a][r][c] = float(T_sp(r, c));
                    sigma_T[i-1][a+n][r][c] = float(T_sp_inv(r, c));
                }  // end c
            }  // end r
        }  // end for a
    }  // end for i
}
