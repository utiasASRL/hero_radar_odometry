#include <steam/steam.hpp>
#include <boost/python/numpy.hpp>
#include "P2P3ErrorEval.hpp"
#include "P2PLandmarkErrorEval.hpp"

namespace p = boost::python;
namespace np = boost::python::numpy;

struct TrajStateVar {
  steam::Time time;
  steam::se3::TransformStateVar::Ptr pose;
  steam::VectorSpaceStateVar::Ptr velocity;
};

char const* run_steam(np::ndarray& p1, np::ndarray& p2, np::ndarray& W, np::ndarray& wij, np::ndarray& T_21) {
//  std::cout << i + j << std::endl;
//  std::cout << "Original array:\n" << p::extract<char const *>(p::str(arr)) << std::endl;
//  arr[1] = 5;
//  std::cout << "Changed array:\n" << p::extract<char const *>(p::str(arr)) << std::endl;
  
  // Number of state times
  unsigned int numPoses = 100;

  // time between states
  double delT = 0.1;

  // Smoothing factor diagonal
  Eigen::Array<double,1,6> Qc_diag;
  Qc_diag << 1.0, 0.001, 0.001, 0.001, 0.001, 0.1;

  // Make Qc_inv
  Eigen::Matrix<double,6,6> Qc_inv; Qc_inv.setZero();
  Qc_inv.diagonal() = 1.0/Qc_diag;

  //
  // Setup initial conditions
  //

  // Pose
  Eigen::Matrix<double,6,1> initPoseVec;
  initPoseVec << 0,0,0,0,0,0;
  lgmath::se3::Transformation initPose(initPoseVec);

  // Zero velocity
  Eigen::Matrix<double,6,1> initVelocity; initVelocity.setZero();

  ///
  /// Setup States
  ///

  // Steam state variables
  std::vector<TrajStateVar> states;

  // Setup state variables - initialized at identity / zero
  for (unsigned int i = 0; i < 2; i++) {
    TrajStateVar temp;
    temp.time = steam::Time(i*delT);
    temp.pose = steam::se3::TransformStateVar::Ptr(new steam::se3::TransformStateVar(initPose));
    temp.velocity = steam::VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(initVelocity));
    states.push_back(temp);
  }

  // Setup Trajectory
  steam::se3::SteamTrajInterface traj(Qc_inv);

  for (unsigned int i = 0; i < states.size(); i++) {
    TrajStateVar& state = states.at(i);
    steam::se3::TransformStateEvaluator::Ptr temp =
        steam::se3::TransformStateEvaluator::MakeShared(state.pose);
    traj.add(state.time, temp, state.velocity);
  }

  ///
  /// Setup Cost Terms
  ///

  // Add priors (alternatively, we could lock pose variable)
  traj.addPosePrior(steam::Time(0.0), initPose, Eigen::Matrix<double,6,6>::Identity());
//  traj.addVelocityPrior(steam::Time(0.0), velocityPrior, Eigen::Matrix<double,6,6>::Identity());

  steam::ParallelizedCostTermCollection::Ptr costTerms(new steam::ParallelizedCostTermCollection());
  traj.appendPriorCostTerms(costTerms);

  // point2point terms
  steam::P2P3ErrorEval::Ptr error;
  steam::WeightedLeastSqCostTerm<3,6>::Ptr cost;
  steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());

  int num_p1 = p1.shape(0);
  int num_p2 = p2.shape(0);
  //  std::cout << num_p1 << ", " << num_p2 << std::endl;
  // std::cout << p::extract<float>(p1[0][0]) << "," << p::extract<float>(p1[0][1]) << std::endl;

  // get T_21
  auto T_21_eval_ptr = traj.getInterpPoseEval(0.1);
  for (int i = 0; i < num_p1; ++i) {

    // get R
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        R(r,c) = double(p::extract<float>(W[i][r][c]));

    // get ref point
    Eigen::Vector4d ref;
    ref << double(p::extract<float>(p1[i][0])), double(p::extract<float>(p1[i][1])), 
           double(p::extract<float>(p1[i][2])), 1.0;

    for (int j = 0; j < num_p2; ++j) {
      // get Rij
      Eigen::Matrix3d Rij = R*double(p::extract<float>(wij[i][j]));
      steam::BaseNoiseModel<3>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<3>(Rij, steam::INFORMATION));

      Eigen::Vector4d read;
      read << double(p::extract<float>(p2[j][0])), double(p::extract<float>(p2[j][1])), 
              double(p::extract<float>(p2[j][2])), 1.0;

      error.reset(new steam::P2P3ErrorEval(ref, read, T_21_eval_ptr));
      cost.reset(new steam::WeightedLeastSqCostTerm<3,6>(error, sharedNoiseModel, sharedLossFunc));
      costTerms->add(cost);
    }
  }
  std::cout << "Done p2p setup" << std::endl;

  ///
  /// Make Optimization Problem
  ///

  // Initialize problem
  steam::OptimizationProblem problem;

  // Add state variables
  for (unsigned int i = 0; i < states.size(); i++) {
    const TrajStateVar& state = states.at(i);
    problem.addStateVariable(state.pose);
    problem.addStateVariable(state.velocity);
  }

  // Add cost terms
  problem.addCostTerm(costTerms);

  ///
  /// Setup Solver and Optimize
  ///

  typedef steam::DoglegGaussNewtonSolver SolverType;

  // Initialize parameters (enable verbose mode)
  SolverType::Params params;
  params.verbose = true;

  // Make solver
  SolverType solver(&problem, params);

  // Optimize
  solver.optimize();

  // Set output
  Eigen::Matrix4d T_21_eigen = states.back().pose->getValue().matrix();
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c)
      T_21[r][c] = float(T_21_eigen(r, c));
  T_21[3][3] = 1.0;

  ///
  /// Print results
  ///

//  std::cout << std::endl
//            << "First Pose: " << states.at(0).pose->getValue()
//            << "Last Pose (full circle): "  << states.back().pose->getValue()
//            << "First Vel: " << states.at(0).velocity->getValue().transpose() << std::endl
//            << "Last Vel:  " << states.back().velocity->getValue().transpose() << std::endl;

  return "hello, world";
}

char const* run_steam_best_match(np::ndarray& p1, np::ndarray& p2, np::ndarray& W, np::ndarray& T_21) {

  // time between states
  double delT = 0.1;

  // Smoothing factor diagonal
  Eigen::Array<double,1,6> Qc_diag;
  Qc_diag << 0.3678912639416186958207788393338,
             0.043068034591947058908889545136844,
             0.1307444996557916849777569723301,
             0.0073124100132336252236275875304727,
             0.0076438703775169331705585662461999,
             0.0021394075786459413462958778495704;

  // Make Qc_inv
  Eigen::Matrix<double,6,6> Qc_inv; Qc_inv.setZero();
  Qc_inv.diagonal() = 1.0/Qc_diag;

  //
  // Setup initial conditions
  //

  // Pose
  Eigen::Matrix<double,6,1> initPoseVec;
  initPoseVec << 0,0,0,0,0,0;
  lgmath::se3::Transformation initPose(initPoseVec);

  // Zero velocity
  Eigen::Matrix<double,6,1> initVelocity; initVelocity.setZero();

  ///
  /// Setup States
  ///

  // Steam state variables
  std::vector<TrajStateVar> states;

  // Setup state variables - initialized at identity / zero
  for (unsigned int i = 0; i < 2; i++) {
    TrajStateVar temp;
    temp.time = steam::Time(i*delT);
    temp.pose = steam::se3::TransformStateVar::Ptr(new steam::se3::TransformStateVar(initPose));
    temp.velocity = steam::VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(initVelocity));
    states.push_back(temp);
  }

  // Setup Trajectory
  steam::se3::SteamTrajInterface traj(Qc_inv);

  for (unsigned int i = 0; i < states.size(); i++) {
    TrajStateVar& state = states.at(i);
    steam::se3::TransformStateEvaluator::Ptr temp =
        steam::se3::TransformStateEvaluator::MakeShared(state.pose);
    traj.add(state.time, temp, state.velocity);
  }

  ///
  /// Setup Cost Terms
  ///

  // Add priors (alternatively, we could lock pose variable)
//  traj.addPosePrior(steam::Time(0.0), initPose, Eigen::Matrix<double,6,6>::Identity());
//  traj.addVelocityPrior(steam::Time(0.0), velocityPrior, Eigen::Matrix<double,6,6>::Identity());

  steam::ParallelizedCostTermCollection::Ptr costTerms(new steam::ParallelizedCostTermCollection());
  traj.appendPriorCostTerms(costTerms);

  // point2point terms
  steam::P2P3ErrorEval::Ptr error;
  steam::WeightedLeastSqCostTerm<3,6>::Ptr cost;
  steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());
  // steam::GemanMcClureLossFunc::Ptr sharedLossFunc(new steam::GemanMcClureLossFunc(1.0));

  int num_p1 = p1.shape(0);
  int num_p2 = p2.shape(0);
  //  std::cout << num_p1 << ", " << num_p2 << std::endl;
  // std::cout << p::extract<float>(p1[0][0]) << "," << p::extract<float>(p1[0][1]) << std::endl;

  // get T_21
  auto T_21_eval_ptr = traj.getInterpPoseEval(0.1);
  for (int i = 0; i < num_p1; ++i) {

    // get R
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        R(r,c) = double(p::extract<float>(W[i][r][c]));
//    R *= double(p::extract<float>(wij[i]));
    steam::BaseNoiseModel<3>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<3>(R, steam::INFORMATION));

    // get ref point
    Eigen::Vector4d ref;
    ref << double(p::extract<float>(p1[i][0])), double(p::extract<float>(p1[i][1])),
           double(p::extract<float>(p1[i][2])), 1.0;

    Eigen::Vector4d read;
    read << double(p::extract<float>(p2[i][0])), double(p::extract<float>(p2[i][1])),
            double(p::extract<float>(p2[i][2])), 1.0;

    error.reset(new steam::P2P3ErrorEval(ref, read, T_21_eval_ptr));
    cost.reset(new steam::WeightedLeastSqCostTerm<3,6>(error, sharedNoiseModel, sharedLossFunc));
    costTerms->add(cost);

  }

  ///
  /// Make Optimization Problem
  ///

  // Initialize problem
  steam::OptimizationProblem problem;

  // Add state variables
//  for (unsigned int i = 0; i < states.size(); i++) {
//    const TrajStateVar& state = states.at(i);
//    problem.addStateVariable(state.pose);
//    problem.addStateVariable(state.velocity);
//  }
  TrajStateVar& state = states.at(0);
//  problem.addStateVariable(state.pose);
  state.pose->setLock(true);
  problem.addStateVariable(state.velocity);
  state = states.at(1);
  problem.addStateVariable(state.pose);
  problem.addStateVariable(state.velocity);

  // Add cost terms
  problem.addCostTerm(costTerms);

  ///
  /// Setup Solver and Optimize
  ///

//  typedef steam::DoglegGaussNewtonSolver SolverType;
  typedef steam::VanillaGaussNewtonSolver SolverType;

  // Initialize parameters (enable verbose mode)
  SolverType::Params params;
  params.verbose = false;

  // Make solver
  SolverType solver(&problem, params);

  // Optimize
  solver.optimize();

  // Set output
  Eigen::Matrix4d T_21_eigen = states.back().pose->getValue().matrix();
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c)
      T_21[0][r][c] = float(T_21_eigen(r, c));
  T_21[0][3][3] = 1.0;

  // sigmapoints
  Eigen::MatrixXd cov = solver.queryCovariance(states.back().pose->getKey());
  Eigen::LLT<Eigen::MatrixXd> lltcov(cov);
  Eigen::MatrixXd L = lltcov.matrixL();
  int n = 6;
  double alpha = sqrt(double(n));
  for (int i = 0; i < n; ++i) {
    Eigen::Matrix4d T_sp = lgmath::se3::vec2tran(L.col(i)*alpha);
    Eigen::Matrix4d T_sp_inv = lgmath::se3::vec2tran(-L.col(i)*alpha);
    T_sp = T_sp*T_21_eigen;
    T_sp_inv = T_sp_inv*T_21_eigen;
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c) {
        T_21[1+i][r][c] = float(T_sp(r, c));
        T_21[1+i+n][r][c] = float(T_sp_inv(r, c));
      } // end c
    T_21[i][3][3] = 1.0;
    T_21[i+n][3][3] = 1.0;
  }  // end i

  ///
  /// Print results
  ///

//  std::cout << std::endl
//            << "First Pose: " << states.at(0).pose->getValue()
//            << "Last Pose (full circle): "  << states.back().pose->getValue()
//            << "First Vel: " << states.at(0).velocity->getValue().transpose() << std::endl
//            << "Last Vel:  " << states.back().velocity->getValue().transpose() << std::endl;

  return "hello, world";
}

char const* run_steam_window(int window_size, np::ndarray& p1, np::ndarray& p2, np::ndarray& W, np::ndarray& T_21) {

  // time between states
  double delT = 0.1;

  // Smoothing factor diagonal
  Eigen::Array<double,1,6> Qc_diag;
  Qc_diag << 0.3678912639416186958207788393338,
             0.043068034591947058908889545136844,
             0.1307444996557916849777569723301,
             0.0073124100132336252236275875304727,
             0.0076438703775169331705585662461999,
             0.0021394075786459413462958778495704;

  // Make Qc_inv
  Eigen::Matrix<double,6,6> Qc_inv; Qc_inv.setZero();
  Qc_inv.diagonal() = 1.0/Qc_diag;

  //
  // Setup initial conditions
  //

  // Pose
  Eigen::Matrix<double,6,1> initPoseVec;
  initPoseVec << 0,0,0,0,0,0;
  lgmath::se3::Transformation initPose(initPoseVec);

  // Zero velocity
  Eigen::Matrix<double,6,1> initVelocity; initVelocity.setZero();

  ///
  /// Setup States
  ///

  // Steam state variables
  std::vector<TrajStateVar> states;

  // Setup state variables - initialized at identity / zero
  for (unsigned int i = 0; i < window_size; i++) {
    TrajStateVar temp;
    temp.time = steam::Time(i*delT);
    temp.pose = steam::se3::TransformStateVar::Ptr(new steam::se3::TransformStateVar(initPose));
    temp.velocity = steam::VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(initVelocity));
    states.push_back(temp);
  }

  // Setup Trajectory
  steam::se3::SteamTrajInterface traj(Qc_inv);

  for (unsigned int i = 0; i < states.size(); i++) {
    TrajStateVar& state = states.at(i);
    steam::se3::TransformStateEvaluator::Ptr temp =
        steam::se3::TransformStateEvaluator::MakeShared(state.pose);
    traj.add(state.time, temp, state.velocity);
    if (i == 0) // lock first pose
      state.pose->setLock(true);
  }

  ///
  /// Setup Cost Terms
  ///

  steam::ParallelizedCostTermCollection::Ptr costTerms(new steam::ParallelizedCostTermCollection());
  traj.appendPriorCostTerms(costTerms);

  // point2point terms
  steam::P2P3ErrorEval::Ptr error;
  steam::WeightedLeastSqCostTerm<3,6>::Ptr cost;
//  steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());
  steam::GemanMcClureLossFunc::Ptr sharedLossFunc(new steam::GemanMcClureLossFunc(1.0));

  int num_p2 = p2.shape(0); // total points
  int npts = p1.shape(0);
  // int num_p2 = p2.shape(0);
  //  std::cout << num_p1 << ", " << num_p2 << std::endl;
  // std::cout << p::extract<float>(p1[0][0]) << "," << p::extract<float>(p1[0][1]) << std::endl;

  // get T_21
  double time_counter = delT;
  int j = 0;
  auto T_21_eval_ptr = traj.getInterpPoseEval(time_counter);
  for (int i = 0; i < num_p2; ++i) {

    // get R
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        R(r,c) = double(p::extract<float>(W[i][r][c]));
//    R *= double(p::extract<float>(wij[i]));
    steam::BaseNoiseModel<3>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<3>(R, steam::INFORMATION));

    // get ref point
    Eigen::Vector4d ref;
    ref << double(p::extract<float>(p1[j][0])), double(p::extract<float>(p1[j][1])),
           double(p::extract<float>(p1[j][2])), 1.0;

    Eigen::Vector4d read;
    read << double(p::extract<float>(p2[i][0])), double(p::extract<float>(p2[i][1])),
            double(p::extract<float>(p2[i][2])), 1.0;

    error.reset(new steam::P2P3ErrorEval(ref, read, T_21_eval_ptr));
    cost.reset(new steam::WeightedLeastSqCostTerm<3,6>(error, sharedNoiseModel, sharedLossFunc));
    costTerms->add(cost);

    // counter
    ++j;
    if (j == npts && i + 1 < num_p2) {
      // reset counter
      j = 0;

      // update T_21
      time_counter += delT;
      auto T_21_eval_ptr = traj.getInterpPoseEval(time_counter);
    }
  }

  ///
  /// Make Optimization Problem
  ///

  // Initialize problem
  steam::OptimizationProblem problem;

  // Add state variables
  for (unsigned int i = 0; i < states.size(); i++) {
    const TrajStateVar& state = states.at(i);
    if (i > 0)
      problem.addStateVariable(state.pose);
    problem.addStateVariable(state.velocity);
  }
  // TrajStateVar& state = states.at(0);
//  problem.addStateVariable(state.pose);
  // state.pose->setLock(true);
  // problem.addStateVariable(state.velocity);
  // state = states.at(1);
  // problem.addStateVariable(state.pose);
  // problem.addStateVariable(state.velocity);

  // Add cost terms
  problem.addCostTerm(costTerms);

  ///
  /// Setup Solver and Optimize
  ///

//  typedef steam::DoglegGaussNewtonSolver SolverType;
  typedef steam::VanillaGaussNewtonSolver SolverType;

  // Initialize parameters (enable verbose mode)
  SolverType::Params params;
  params.verbose = false;

  // Make solver
  SolverType solver(&problem, params);

  // Optimize
  solver.optimize();

  // Set output
  for (int window = 1; window < window_size; ++window) {
    const TrajStateVar& state = states.at(window);

    Eigen::Matrix4d T_21_eigen = state.pose->getValue().matrix();
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c)
        T_21[window-1][0][r][c] = float(T_21_eigen(r, c));
    T_21[window-1][0][3][3] = 1.0;

    // sigmapoints
    Eigen::MatrixXd cov = solver.queryCovariance(state.pose->getKey());
    Eigen::LLT<Eigen::MatrixXd> lltcov(cov);
    Eigen::MatrixXd L = lltcov.matrixL();
    int n = 6;
    double alpha = sqrt(double(n));
    for (int i = 0; i < n; ++i) {
      Eigen::Matrix4d T_sp = lgmath::se3::vec2tran(L.col(i)*alpha);
      Eigen::Matrix4d T_sp_inv = lgmath::se3::vec2tran(-L.col(i)*alpha);
      T_sp = T_sp*T_21_eigen;
      T_sp_inv = T_sp_inv*T_21_eigen;
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
          T_21[window-1][1+i][r][c] = float(T_sp(r, c));
          T_21[window-1][1+i+n][r][c] = float(T_sp_inv(r, c));
        } // end c
      T_21[window-1][i][3][3] = 1.0;
      T_21[window-1][i+n][3][3] = 1.0;
    }  // end i
  }   // end window

  ///
  /// Print results
  ///

//  std::cout << std::endl
//            << "First Pose: " << states.at(0).pose->getValue()
//            << "Last Pose (full circle): "  << states.back().pose->getValue()
//            << "First Vel: " << states.at(0).velocity->getValue().transpose() << std::endl
//            << "Last Vel:  " << states.back().velocity->getValue().transpose() << std::endl;

  return "hello, world";
}

char const* run_steam_window_lm(bool compute_sp, int window_size, np::ndarray& mask, np::ndarray& p1, np::ndarray& p2, np::ndarray& W, np::ndarray& T_21, np::ndarray& l_sp) {

  // time between states
  double delT = 0.1;

  // Smoothing factor diagonal
  Eigen::Array<double,1,6> Qc_diag;
  Qc_diag << 0.3678912639416186958207788393338,
             0.043068034591947058908889545136844,
             0.1307444996557916849777569723301,
             0.0073124100132336252236275875304727,
             0.0076438703775169331705585662461999,
             0.0021394075786459413462958778495704;

  // Make Qc_inv
  Eigen::Matrix<double,6,6> Qc_inv; Qc_inv.setZero();
  Qc_inv.diagonal() = 1.0/Qc_diag;

  //
  // Setup initial conditions
  //

  // Pose
  Eigen::Matrix<double,6,1> initPoseVec;
  initPoseVec << 0,0,0,0,0,0;
  lgmath::se3::Transformation initPose(initPoseVec);

  // Zero velocity
  Eigen::Matrix<double,6,1> initVelocity; initVelocity.setZero();

  ///
  /// Setup States
  ///

  // Steam state variables
  std::vector<TrajStateVar> states;
  std::vector<steam::se3::LandmarkStateVar::Ptr> landmarks;

  // Setup state variables - initialized at identity / zero
  for (unsigned int i = 0; i < window_size; i++) {
    TrajStateVar temp;
    temp.time = steam::Time(i*delT);
    temp.pose = steam::se3::TransformStateVar::Ptr(new steam::se3::TransformStateVar(initPose));
    temp.velocity = steam::VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(initVelocity));
    states.push_back(temp);
  }

  // Setup landmarks initialized with p1 (technically can be wrong frame)
  int npts = p1.shape(0);
  landmarks.resize(npts);
  for (unsigned int i = 0; i < npts; i++) {
      // Insert the landmark initialized to first frame observation
    Eigen::Vector3d temp_lm;
    temp_lm << double(p::extract<float>(p1[i][0])), double(p::extract<float>(p1[i][1])),
               double(p::extract<float>(p1[i][2]));
    landmarks[i] = steam::se3::LandmarkStateVar::Ptr(new steam::se3::LandmarkStateVar(temp_lm));
      // landmarks[i] = steam::se3::LandmarkStateVar::Ptr(new steam::se3::LandmarkStateVar(landmarks_gt[i]));
  }

  // Setup Trajectory
  steam::se3::SteamTrajInterface traj(Qc_inv);

  for (unsigned int i = 0; i < states.size(); i++) {
    TrajStateVar& state = states.at(i);
    steam::se3::TransformStateEvaluator::Ptr temp =
        steam::se3::TransformStateEvaluator::MakeShared(state.pose);
    traj.add(state.time, temp, state.velocity);
    if (i == 0) // lock first pose
      state.pose->setLock(true);
  }

  ///
  /// Setup Cost Terms
  ///

  steam::ParallelizedCostTermCollection::Ptr costTerms(new steam::ParallelizedCostTermCollection());
  traj.appendPriorCostTerms(costTerms);

  // point2point terms
  // steam::P2PLandmarkErrorEval::Ptr error;
  // steam::WeightedLeastSqCostTerm<3,6>::Ptr cost;
//  steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());
  steam::GemanMcClureLossFunc::Ptr sharedLossFunc(new steam::GemanMcClureLossFunc(1.0));

  int num_p2 = p2.shape(0); // total points
  // int num_p2 = p2.shape(0);
  //  std::cout << num_p1 << ", " << num_p2 << std::endl;
  // std::cout << p::extract<float>(p1[0][0]) << "," << p::extract<float>(p1[0][1]) << std::endl;

  // get T_21
  double time_counter = 0.0;
  int j = 0;  // landmark counter
  int k = 0;  // window counter
  auto T_21_eval_ptr = traj.getInterpPoseEval(time_counter);
  // std::cout << "start cost terms" << std::endl;
  // std::cout << num_p2 << std::endl;
  for (int i = 0; i < num_p2; ++i) {
    // check mask
    if (p::extract<bool>(mask[k][j])) {

      // get R
      Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
          R(r,c) = double(p::extract<float>(W[i][r][c]));

      steam::BaseNoiseModel<3>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<3>(R, steam::INFORMATION));

      // get ref point
      // Eigen::Vector4d ref;
      // ref << double(p::extract<float>(p1[j][0])), double(p::extract<float>(p1[j][1])),
      //        double(p::extract<float>(p1[j][2])), 1.0;

      Eigen::Vector4d read;
      read << double(p::extract<float>(p2[i][0])), double(p::extract<float>(p2[i][1])),
              double(p::extract<float>(p2[i][2])), 1.0;

      // error.reset(new steam::P2PLandmarkErrorEval(landmarks[j], read, T_21_eval_ptr));
      steam::P2PLandmarkErrorEval::Ptr error(new steam::P2PLandmarkErrorEval(landmarks[j], read, T_21_eval_ptr));
            // steam::P2PLandmarkErrorEval::Ptr errorfunc(new steam::P2PLandmarkErrorEval(
            //   landmarks[j], homog_meas, T_k0_eval_ptr));
      // cost.reset(new steam::WeightedLeastSqCostTerm<3,6>(error, sharedNoiseModel, sharedLossFunc));
      steam::WeightedLeastSqCostTerm<3,6>::Ptr cost(
          new steam::WeightedLeastSqCostTerm<3,6>(error, sharedNoiseModel, sharedLossFunc));
      costTerms->add(cost);
    }
    // else{
    //   std::cout << i << std::endl;
    // }

    // counter
    ++j;
    if (j == npts && i + 1 < num_p2) {
    // if (j == npts && k < window_size - 1) {
      // reset counter
      j = 0;

      // update T_21
      time_counter += delT;
      T_21_eval_ptr = traj.getInterpPoseEval(time_counter);
      ++k;
    }
  }
  // std::cout << j << "," << k << std::endl;
  // std::cout << "create cost terms" << std::endl;
  ///
  /// Make Optimization Problem
  ///

  // Initialize problem
  steam::OptimizationProblem problem;

  // Add state variables
  for (unsigned int i = 0; i < states.size(); i++) {
    const TrajStateVar& state = states.at(i);
    if (i > 0)
      problem.addStateVariable(state.pose);
    problem.addStateVariable(state.velocity);
  }

  // Add landmark variables
  for (unsigned int i = 0; i < landmarks.size(); i++) {
    problem.addStateVariable(landmarks[i]);
  }

  // Add cost terms
  problem.addCostTerm(costTerms);

  ///
  /// Setup Solver and Optimize
  ///

//  typedef steam::DoglegGaussNewtonSolver SolverType;
  typedef steam::VanillaGaussNewtonSolver SolverType;

  // Initialize parameters (enable verbose mode)
  SolverType::Params params;
  params.verbose = false;

  // Make solver
  SolverType solver(&problem, params);

  // Optimize
  solver.optimize();

  // Set output
  for (int window = 0; window < window_size; ++window) {
    if (window == 0) {
      // only landmark sigmapoints for first
      for (int r = 0; r < 4; ++r)
        T_21[window][r][r] = 1.0;

      // compute sigmapoints?
      if (!compute_sp)
        continue;

      // loop over landmarks
      for (int l = 0; l < landmarks.size(); ++l) {
        if (p::extract<bool>(mask[window][l])) {
          const steam::se3::LandmarkStateVar::Ptr& landmark = landmarks.at(l);
          Eigen::Vector3d mean = (landmark->getValue()/landmark->getValue()(3)).head<3>();
          l_sp[window][0][l][0] = float(mean(0));
          l_sp[window][0][l][1] = float(mean(1));
          l_sp[window][0][l][2] = float(mean(2));
          Eigen::MatrixXd cov = solver.queryCovariance(landmark->getKey());
          Eigen::LLT<Eigen::MatrixXd> lltcov(cov);
          Eigen::MatrixXd L = lltcov.matrixL();
          int n = 3;
          double alpha = sqrt(double(n));
          for (int i = 0; i < n; ++i) {
            Eigen::Vector3d delta = L.col(i)*alpha;
            Eigen::Vector3d psp = mean + delta;
            Eigen::Vector3d nsp = mean - delta;
            for (int r = 0; r < 3; ++r) {
                l_sp[window][1+i][l][r] = float(psp(r));
                l_sp[window][1+i+n][l][r] = float(nsp(r));
            } // end r
          }  // end i

        }
      } // end l

      continue;
    } // end if window == 0

    const TrajStateVar& state = states.at(window);

    // set mean transform
    Eigen::Matrix4d T_21_eigen = state.pose->getValue().matrix();
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c)
        T_21[window][r][c] = float(T_21_eigen(r, c));
    T_21[window][3][3] = 1.0;

    // compute sigmapoints?
    if (!compute_sp)
      continue;

    // sigmapoints for each landmark
    for (int l = 0; l < landmarks.size(); ++l) {
      // skip if no observation in this window frame
      if (!p::extract<bool>(mask[window][l]))
        continue;

      // mean landmark
      const steam::se3::LandmarkStateVar::Ptr& landmark = landmarks.at(l);
      Eigen::Vector3d mean_lm = (landmark->getValue()/landmark->getValue()(3)).head<3>();
      Eigen::Vector3d mean_lm_tf = T_21_eigen.topLeftCorner(3,3)*mean_lm + T_21_eigen.topRightCorner(3,1);
      l_sp[window][0][l][0] = float(mean_lm_tf(0));
      l_sp[window][0][l][1] = float(mean_lm_tf(1));
      l_sp[window][0][l][2] = float(mean_lm_tf(2));

      // state keys (tf + landmark)
      std::vector<steam::StateKey> keys(2);
      keys.at(0) = state.pose->getKey();
      keys.at(1) = landmark->getKey();

      // get covariance and LLT decomp
      steam::BlockMatrix cov_blocks = solver.queryCovarianceBlock(keys);
      Eigen::Matrix<double, 9, 9> cov;
      cov.topLeftCorner(6,6) = cov_blocks.at(0, 0);
      cov.bottomLeftCorner(3,6) = cov_blocks.at(1, 0);
      cov.topRightCorner(6,3) = cov_blocks.at(1, 0).transpose();
      cov.bottomRightCorner(3,3) = cov_blocks.at(1, 1);
      Eigen::LLT<Eigen::MatrixXd> lltcov(cov);
      Eigen::MatrixXd L = lltcov.matrixL();

      int n = 9;  // 6 pose + 3 landmark
      double alpha = sqrt(double(n));
      for (int i = 0; i < n; ++i) {
        // delta for pose
        Eigen::Matrix4d T_sp = lgmath::se3::vec2tran(L.col(i).head<6>()*alpha);
        Eigen::Matrix4d T_sp_inv = lgmath::se3::vec2tran(-L.col(i).head<6>()*alpha);

        // delta for landmark
        Eigen::Vector3d lm_delta = L.col(i).tail<3>()*alpha;
        
        // positive/negative sigmapoints
        T_sp = T_sp*T_21_eigen;
        T_sp_inv = T_sp_inv*T_21_eigen;
        Eigen::Vector3d psp = mean_lm + lm_delta;
        Eigen::Vector3d nsp = mean_lm - lm_delta;
        psp = T_sp.topLeftCorner(3,3)*psp + T_sp.topRightCorner(3,1);
        nsp = T_sp_inv.topLeftCorner(3,3)*nsp + T_sp_inv.topRightCorner(3,1);

        // set output
        for (int c = 0; c < 3; ++c) {
          l_sp[window][1+i][l][c] = float(psp(c));
          l_sp[window][1+i+n][l][c] = float(nsp(c));
        } // end c
      }  // end i
    } // end l
  } // end window

  ///
  /// Print results
  ///

//  std::cout << std::endl
//            << "First Pose: " << states.at(0).pose->getValue()
//            << "Last Pose (full circle): "  << states.back().pose->getValue()
//            << "First Vel: " << states.at(0).velocity->getValue().transpose() << std::endl
//            << "Last Vel:  " << states.back().velocity->getValue().transpose() << std::endl;

  return "hello, world";
}

char const* run_steam_no_prior(np::ndarray& p1, np::ndarray& p2, np::ndarray& W, np::ndarray& wij, np::ndarray& T_21) {

  // time between states
  double delT = 0.1;

  // Smoothing factor diagonal
  Eigen::Array<double,1,6> Qc_diag;
  Qc_diag << 1.0, 0.001, 0.001, 0.001, 0.001, 0.1;

  // Make Qc_inv
  Eigen::Matrix<double,6,6> Qc_inv; Qc_inv.setZero();
  Qc_inv.diagonal() = 1.0/Qc_diag;

  //
  // Setup initial conditions
  //

  // Pose
  Eigen::Matrix<double,6,1> initPoseVec;
  initPoseVec << 0,0,0,0,0,0;
  lgmath::se3::Transformation initPose(initPoseVec);

  // Zero velocity
  Eigen::Matrix<double,6,1> initVelocity; initVelocity.setZero();

  ///
  /// Setup States
  ///

  // Steam state variables
  std::vector<TrajStateVar> states;

  // Setup state variables - initialized at identity / zero
  for (unsigned int i = 0; i < 2; i++) {
    TrajStateVar temp;
    temp.time = steam::Time(i*delT);
    temp.pose = steam::se3::TransformStateVar::Ptr(new steam::se3::TransformStateVar(initPose));
    temp.velocity = steam::VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(initVelocity));
    states.push_back(temp);
  }

  // Setup Trajectory
  steam::se3::SteamTrajInterface traj(Qc_inv);

  for (unsigned int i = 0; i < states.size(); i++) {
    TrajStateVar& state = states.at(i);
    steam::se3::TransformStateEvaluator::Ptr temp =
        steam::se3::TransformStateEvaluator::MakeShared(state.pose);
    traj.add(state.time, temp, state.velocity);
  }

  ///
  /// Setup Cost Terms
  ///

  // Add priors (alternatively, we could lock pose variable)
//  traj.addPosePrior(steam::Time(0.0), initPose, Eigen::Matrix<double,6,6>::Identity());
//  traj.addVelocityPrior(steam::Time(0.0), velocityPrior, Eigen::Matrix<double,6,6>::Identity());

  steam::ParallelizedCostTermCollection::Ptr costTerms(new steam::ParallelizedCostTermCollection());
  // traj.appendPriorCostTerms(costTerms);

  // point2point terms
  steam::P2P3ErrorEval::Ptr error;
  steam::WeightedLeastSqCostTerm<3,6>::Ptr cost;
  // steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());
  steam::GemanMcClureLossFunc::Ptr sharedLossFunc(new steam::GemanMcClureLossFunc(1.0));

  int num_p1 = p1.shape(0);
  int num_p2 = p2.shape(0);
  //  std::cout << num_p1 << ", " << num_p2 << std::endl;
  // std::cout << p::extract<float>(p1[0][0]) << "," << p::extract<float>(p1[0][1]) << std::endl;

  // get T_21
  auto T_21_eval_ptr = traj.getInterpPoseEval(0.1);
  for (int i = 0; i < num_p1; ++i) {

    // get R
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        R(r,c) = double(p::extract<float>(W[i][r][c]));
    R *= double(p::extract<float>(wij[i]));
    steam::BaseNoiseModel<3>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<3>(R, steam::INFORMATION));

    // get ref point
    Eigen::Vector4d ref;
    ref << double(p::extract<float>(p1[i][0])), double(p::extract<float>(p1[i][1])),
           double(p::extract<float>(p1[i][2])), 1.0;

    Eigen::Vector4d read;
    read << double(p::extract<float>(p2[i][0])), double(p::extract<float>(p2[i][1])),
            double(p::extract<float>(p2[i][2])), 1.0;

    error.reset(new steam::P2P3ErrorEval(ref, read, T_21_eval_ptr));
    cost.reset(new steam::WeightedLeastSqCostTerm<3,6>(error, sharedNoiseModel, sharedLossFunc));
    costTerms->add(cost);

  }

  ///
  /// Make Optimization Problem
  ///

  // Initialize problem
  steam::OptimizationProblem problem;

  // Add state variables
//  for (unsigned int i = 0; i < states.size(); i++) {
//    const TrajStateVar& state = states.at(i);
//    problem.addStateVariable(state.pose);
//    problem.addStateVariable(state.velocity);
//  }
  const TrajStateVar& state = states.at(1);
  problem.addStateVariable(state.pose);

  // Add cost terms
  problem.addCostTerm(costTerms);

  ///
  /// Setup Solver and Optimize
  ///

//  typedef steam::DoglegGaussNewtonSolver SolverType;
  typedef steam::VanillaGaussNewtonSolver SolverType;

  // Initialize parameters (enable verbose mode)
  SolverType::Params params;
  params.verbose = false;

  // Make solver
  SolverType solver(&problem, params);

  // Optimize
  solver.optimize();

  // Set output
  Eigen::Matrix4d T_21_eigen = states.back().pose->getValue().matrix();
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c)
      T_21[r][c] = float(T_21_eigen(r, c));
  T_21[3][3] = 1.0;

  ///
  /// Print results
  ///

//  std::cout << std::endl
//            << "First Pose: " << states.at(0).pose->getValue()
//            << "Last Pose (full circle): "  << states.back().pose->getValue()
//            << "First Vel: " << states.at(0).velocity->getValue().transpose() << std::endl
//            << "Last Vel:  " << states.back().velocity->getValue().transpose() << std::endl;

  return "hello, world";
}

#include <boost/python.hpp>

BOOST_PYTHON_MODULE(steampy_f2f)
{
    Py_Initialize();
    np::initialize();
//    using namespace boost::python;
    p::def("run_steam", run_steam);
    p::def("run_steam_best_match", run_steam_best_match);
    p::def("run_steam_window", run_steam_window);
    p::def("run_steam_window_lm", run_steam_window_lm);
    p::def("run_steam_no_prior", run_steam_no_prior);
}
