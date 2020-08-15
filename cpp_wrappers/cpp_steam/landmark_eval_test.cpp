//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SimpleBAandTrajPrior.cpp
/// \brief A sample usage of the STEAM Engine library for a bundle adjustment problem
///        with relative landmarks and trajectory smoothing factors.
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <random>
#include <lgmath.hpp>
#include <steam.hpp>
#include "P2PLandmarkErrorEval.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Structure to store trajectory state variables
//////////////////////////////////////////////////////////////////////////////////////////////
struct TrajStateVar {
  steam::Time time;
  steam::se3::TransformStateVar::Ptr pose;
  steam::VectorSpaceStateVar::Ptr velocity;
};

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Example that loads and solves simple bundle adjustment problems
//////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

  ///
  /// Setup Traj Prior
  ///


  // Smoothing factor diagonal -- in this example, we penalize accelerations in each dimension
  //                              except for the forward and yaw (this should be fairly typical)
  double lin_acc_stddev_x = 1.00; // body-centric (e.g. x is usually forward)
  double lin_acc_stddev_y = 0.01; // body-centric (e.g. y is usually side-slip)
  double lin_acc_stddev_z = 0.01; // body-centric (e.g. z is usually 'jump')
  double ang_acc_stddev_x = 0.01; // ~roll
  double ang_acc_stddev_y = 0.01; // ~pitch
  double ang_acc_stddev_z = 1.00; // ~yaw
  Eigen::Array<double,1,6> Qc_diag;
  Qc_diag << lin_acc_stddev_x, lin_acc_stddev_y, lin_acc_stddev_z,
             ang_acc_stddev_x, ang_acc_stddev_y, ang_acc_stddev_z;

  // Make Qc_inv
  Eigen::Matrix<double,6,6> Qc_inv; Qc_inv.setZero();
    Qc_inv.diagonal() = 1.0/Qc_diag;

  ///
  /// Create Dataset
  ///

  // Ground truth
  int window_size = 2;
  int landmarks_total = 100;
  std::default_random_engine gen;
  std::uniform_real_distribution<double> dist(-30.0, 30.0);
  std::vector<lgmath::se3::Transformation> poses_gt_k_0;
  std::vector<Eigen::Vector3d> landmarks_gt;
  std::cout << "Creating groundtruth..." << std::endl;

  // landmarks
  for (int i = 0; i < landmarks_total; ++i) {
    Eigen::Vector3d lm_position(dist(gen), dist(gen), dist(gen));
    // std::cout << lm_position.transpose() << std::endl;
    // steam::se3::LandmarkStateVar::Ptr temp(new steam::se3::LandmarkStateVar(lm_position));
    landmarks_gt.push_back(lm_position);
    // std::cout << "New landmark: " << landmarks_gt.back()->getValue().head<3>().transpose()/
    //   landmarks_gt.back()->getValue()(3) << std::endl;
    std::cout << "New landmark: " << landmarks_gt.back().transpose() << std::endl;
  }   // end i

  // poses
  // {
    double v_x = -1.0;
    double omega_z = 0.05;
    Eigen::Matrix<double,6,1> measVec;
    measVec << v_x, 0.0, 0.0, 0.0, 0.0, omega_z;
    lgmath::se3::Transformation compose_pose;
    for (int i = 0; i < window_size; ++i) {
      poses_gt_k_0.push_back(compose_pose);
      compose_pose = lgmath::se3::Transformation(measVec)*compose_pose;
      // std::cout << "New pose: " << poses_gt_k_0.back().inverse().matrix().col(3).head<3>().transpose() << std::endl;
      std::cout << "New pose: " << std::endl;;
      std::cout << poses_gt_k_0.back().matrix() << std::endl;
    } // end i
  // }

  ///
  /// Setup Measurements
  ///
  std::vector<std::vector<Eigen::Vector3d>> measurements;
  std::normal_distribution<double> ndist(0.0, 0.5);
  for (int i = 0; i < window_size; ++i) {
    // pose
    lgmath::se3::Transformation& curr_pose = poses_gt_k_0[i];

    // meas window
    std::vector<Eigen::Vector3d> curr_meas;
    for (int j = 0; j < landmarks_total; ++j) {
      // convert gt landmark to local frame
      Eigen::Vector3d noise(ndist(gen), ndist(gen), ndist(gen));
      Eigen::Vector4d local_landmark = curr_pose.matrix()
        *Eigen::Vector4d(landmarks_gt[j](0), landmarks_gt[j](1), landmarks_gt[j](2), 1);
      Eigen::Vector3d meas = noise + local_landmark.head<3>();
      // Eigen::Vector3d meas = local_landmark.head<3>();
      // std::cout << "Landmark "<< j << " : " << meas.transpose() << std::endl;
      curr_meas.push_back(meas);
    }
    measurements.push_back(curr_meas);
  }   // end i

  ///
  /// Setup States
  ///

  // Set a fixed identity transform that will be used to initialize landmarks in their parent frame
  steam::se3::FixedTransformEvaluator::Ptr tf_identity =
      steam::se3::FixedTransformEvaluator::MakeShared(lgmath::se3::Transformation());


  // State variable containers (and related data)
  std::vector<TrajStateVar> traj_states;
  std::vector<steam::se3::LandmarkStateVar::Ptr> landmarks;

  ///
  /// Initialize States
  ///

  // Zero velocity
  Eigen::Matrix<double,6,1> initVelocity; initVelocity.setZero();

  // Setup state variables using initial condition
  double time = 0.0;
  double dt = 1.0;
  for (unsigned int i = 0; i < window_size; i++) {
    TrajStateVar temp;
    temp.time = steam::Time(time);
    temp.pose = steam::se3::TransformStateVar::Ptr(new steam::se3::TransformStateVar());
    // temp.pose = steam::se3::TransformStateVar::Ptr(new steam::se3::TransformStateVar(poses_gt_k_0[i]));
    // temp.velocity = steam::VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(initVelocity));
    temp.velocity = steam::VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(measVec));
    traj_states.push_back(temp);
    time += dt;
  }

  // Setup Trajectory
  steam::se3::SteamTrajInterface traj(Qc_inv);
  for (unsigned int i = 0; i < traj_states.size(); i++) {
    TrajStateVar& state = traj_states.at(i);
    steam::se3::TransformStateEvaluator::Ptr temp =
        steam::se3::TransformStateEvaluator::MakeShared(state.pose);
    traj.add(state.time, temp, state.velocity);
  }

  // Lock first pose (otherwise entire solution is 'floating')
  //  **Note: alternatively we could add a prior to the first pose.
  traj_states[0].pose->setLock(true);
  // traj_states[0].velocity->setLock(true);

  // Setup landmarks
  landmarks.resize(landmarks_total);
  for (unsigned int i = 0; i < landmarks_total; i++) {
      // Insert the landmark initialized to first frame observation
      landmarks[i] = steam::se3::LandmarkStateVar::Ptr(new steam::se3::LandmarkStateVar(measurements[0][i]));
      // landmarks[i] = steam::se3::LandmarkStateVar::Ptr(new steam::se3::LandmarkStateVar(landmarks_gt[i]));
  }

  ///
  /// Setup Cost Terms
  ///

  // steam cost terms
  steam::ParallelizedCostTermCollection::Ptr measCostTerms(new steam::ParallelizedCostTermCollection());

  // Setup shared noise and loss function
  Eigen::Matrix3d noise_matrix = Eigen::Matrix3d::Identity()*0.5*0.5;
  steam::BaseNoiseModel<3>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<3>(noise_matrix));
  // steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());
  steam::GemanMcClureLossFunc::Ptr sharedLossFunc(new steam::GemanMcClureLossFunc(1.0));

  // Generate cost terms for camera measurements
  time = 0.0;
  for (unsigned int i = 0; i < window_size; ++i) {
    // get pose
    auto T_k0_eval_ptr = traj.getInterpPoseEval(time);
    // std::cout << T_k0_eval_ptr->evaluate().matrix() << std::endl;
    for (unsigned int j = 0; j < landmarks_total; ++j) {

      // Construct error function
        // P2PLandmarkErrorEval(const se3::LandmarkStateVar::Ptr& ref_a,
        //         const Eigen::Vector4d& read_b,
        //         const se3::TransformEvaluator::ConstPtr& T_b_a
        //         );
      Eigen::Vector4d homog_meas;
      homog_meas << measurements[i][j](0), measurements[i][j](1), measurements[i][j](2), 1.0;
      // std::cout << homog_meas.transpose() << std::endl;
      steam::P2PLandmarkErrorEval::Ptr errorfunc(new steam::P2PLandmarkErrorEval(
              landmarks[j], homog_meas, T_k0_eval_ptr));

      // Construct cost term
      steam::WeightedLeastSqCostTerm<3,6>::Ptr cost(
        new steam::WeightedLeastSqCostTerm<3,6>(errorfunc, sharedNoiseModel, sharedLossFunc));
      measCostTerms->add(cost);
    }
    time += dt;
  }

  // Trajectory prior smoothing terms
  steam::ParallelizedCostTermCollection::Ptr smoothingCostTerms(new steam::ParallelizedCostTermCollection());
  traj.appendPriorCostTerms(smoothingCostTerms);

  ///
  /// Make Optimization Problem
  ///

  // Initialize problem
  steam::OptimizationProblem problem;

  // Add state variables
  for (unsigned int i = 0; i < traj_states.size(); i++) {
    const TrajStateVar& state = traj_states.at(i);
    if (i > 0) {
      problem.addStateVariable(state.pose);
    }
    problem.addStateVariable(state.velocity);
  }

  // Add landmark variables
  for (unsigned int i = 0; i < landmarks.size(); i++) {
    problem.addStateVariable(landmarks[i]);
  }

  // Add cost terms
  problem.addCostTerm(measCostTerms);
  problem.addCostTerm(smoothingCostTerms);

  ///
  /// Setup Solver and Optimize
  ///
  // typedef steam::DoglegGaussNewtonSolver SolverType;
  typedef steam::VanillaGaussNewtonSolver SolverType;

  // Initialize parameters (enable verbose mode)
  SolverType::Params params;
  params.verbose = true;

  // Make solver
  SolverType solver(&problem, params);

  // Optimize
  solver.optimize();

  // Setup Trajectory
  for (unsigned int i = 0; i < traj_states.size(); i++) {
    TrajStateVar& state = traj_states.at(i);
    std::cout << i << ": \n " << state.pose->getValue() << "\n";
  }

  // const std::vector<steam::StateKey>& keys
  // Eigen::MatrixXd cov = solver.queryCovariance(state.pose->getKey());
  // Eigen::LLT<Eigen::MatrixXd> lltcov(cov);
  // Eigen::MatrixXd L = lltcov.matrixL();

  // std::vector<steam::StateKey> keys(2);
  // keys.at(0) = traj_states.at(1).pose->getKey();
  // keys.at(1) = landmarks.at(0)->getKey();
  // steam::BlockMatrix cov_blocks = solver.queryCovarianceBlock(keys);
  // std::cout << cov_blocks.at(0, 0) << std::endl << std::endl;
  // std::cout << cov_blocks.at(1, 0) << std::endl << std::endl;
  // std::cout << cov_blocks.at(1, 1) << std::endl << std::endl;
  // Eigen::Matrix<double, 9, 9> cov;
  // cov.topLeftCorner(6,6) = cov_blocks.at(0, 0);
  // cov.bottomLeftCorner(3,6) = cov_blocks.at(1, 0);
  // cov.topRightCorner(6,3) = cov_blocks.at(1, 0).transpose();
  // cov.bottomRightCorner(3,3) = cov_blocks.at(1, 1);
  // std::cout << cov << std::endl << std::endl;
  // std::cout << "mean landmark: " << (landmarks.at(0)->getValue()/landmarks.at(0)->getValue()(3)).transpose() << std::endl;

  return 0;
}
