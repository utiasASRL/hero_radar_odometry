#include "SteamSolver.hpp"
#include "P2P3ErrorEval.hpp"

// Reset trajectory to identity poses and zero velocities
void SteamSolver::resetTraj() {

  Eigen::Matrix<double,4,4> eig_identity = Eigen::Matrix<double,4,4>::Identity();
  lgmath::se3::Transformation T_identity(eig_identity);
  Eigen::Matrix<double,6,1> zero_vel;
  zero_vel.setZero();

  states_.clear();
  states_.reserve(window_size_);
  for (unsigned int k = 0; k < window_size_; ++k) {

    TrajStateVar temp;
    temp.time = steam::Time(k*dt_);
    temp.pose = steam::se3::TransformStateVar::Ptr(new steam::se3::TransformStateVar(T_identity));
    temp.velocity = steam::VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(zero_vel));
    states_.push_back(temp);
  }
}

// Set the Qc inverse matrix with the diagonal of Qc
void SteamSolver::setQcInv(const np::ndarray& Qc_diag) {
  Eigen::Matrix<double,6,1> temp = numpyToEigen2D(Qc_diag);
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
  for (unsigned int i = 0; i < states_.size(); i++) {
    TrajStateVar& state = states_.at(i);
    steam::se3::TransformStateEvaluator::Ptr temp =
        steam::se3::TransformStateEvaluator::MakeShared(state.pose);
    traj.add(state.time, temp, state.velocity);
    if (i == 0) // lock first pose
      state.pose->setLock(true);
  } // end i


  // Cost Terms
  steam::ParallelizedCostTermCollection::Ptr costTerms(new steam::ParallelizedCostTermCollection());
  traj.appendPriorCostTerms(costTerms);

  // steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());
  steam::GemanMcClureLossFunc::Ptr sharedLossFunc(new steam::GemanMcClureLossFunc(1.0));

  // loop through every frame
  for (unsigned int i = 1; i < window_size_; i++) {
    // get pose
    auto T_k0_eval_ptr = traj.getInterpPoseEval(steam::Time(i*dt_));

    unsigned int num_meas = p2_[i-1].shape(0);
    for (unsigned int j = 0; j < num_meas; ++j) {

      // get R
      Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
          R(r,c) = p::extract<float>(w_[i-1][j][r][c]);

      steam::BaseNoiseModel<3>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<3>(R, steam::INFORMATION));

      // get measurement
      Eigen::Vector4d read;
      read << double(p::extract<float>(p2_[i-1][j][0])), double(p::extract<float>(p2_[i-1][j][1])),
              double(p::extract<float>(p2_[i-1][j][2])), 1.0;

      Eigen::Vector4d ref;
      ref << double(p::extract<float>(p1_[i-1][j][0])), double(p::extract<float>(p1_[i-1][j][1])),
             double(p::extract<float>(p1_[i-1][j][2])), 1.0;

      steam::P2P3ErrorEval::Ptr error(new steam::P2P3ErrorEval(ref, read, T_k0_eval_ptr));
      steam::WeightedLeastSqCostTerm<3,6>::Ptr cost(
          new steam::WeightedLeastSqCostTerm<3,6>(error, sharedNoiseModel, sharedLossFunc));
      costTerms->add(cost);
    } // end j
  } // end i

  // Initialize problem
  steam::OptimizationProblem problem;

  // Add state variables
  for (unsigned int i = 0; i < states_.size(); i++) {
    const TrajStateVar& state = states_.at(i);
    problem.addStateVariable(state.pose);
    problem.addStateVariable(state.velocity);
  }

  // Add cost terms
  problem.addCostTerm(costTerms);

  // Solver parameters
  // TODO: Make this a parameter
  // typedef steam::DoglegGaussNewtonSolver SolverType;
  // typedef steam::LevMarqGaussNewtonSolver SolverType;
  typedef steam::VanillaGaussNewtonSolver SolverType;
  SolverType::Params params;
  params.verbose = false; // TODO: make this a parameter

  // Make solver
  solver_ = SolverBasePtr(new SolverType(&problem, params));

  // Optimize
  solver_->optimize();
}

void SteamSolver::getPoses(np::ndarray& poses) {
  for (int i = 0; i < states_.size(); ++i) {
    // get position
    Eigen::Matrix<double,4,4> Tvi = states_[i].pose->getValue().matrix();

    // set output
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 4; ++c)
        poses[i][r][c] = float(Tvi(r,c));
  }
}

void SteamSolver::getVelocities(np::ndarray& vels) {
  for (int i = 0; i < states_.size(); ++i) {
    // get position
    Eigen::Matrix<double,6,1> vel = states_[i].velocity->getValue();

    // set output
    for (int r = 0; r < 6; ++r)
      vels[i][r] = float(vel(r));
  }
}
