#include <steam/steam.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "P2P3ErrorEval.hpp"
#include "P2PLandmarkErrorEval.hpp"

namespace p = boost::python;
namespace np = boost::python::numpy;

// convert python list to std::vector
template< typename T >
inline
std::vector< T > to_std_vector( const p::object& iterable ) {
  return std::vector< T >( p::stl_input_iterator< T >( iterable ),
                           p::stl_input_iterator< T >( ) );
}

// trajectory state struct
struct TrajStateVar {
  steam::Time time;
  steam::se3::TransformStateVar::Ptr pose;
  steam::VectorSpaceStateVar::Ptr velocity;
};

// run steam optimization
void run_steam(const p::object& p2_list, const p::object& p1_list, const p::object& weight_list,
    np::ndarray& poses, np::ndarray& vels, bool compute_sp, int delTmult, double dt, np::ndarray& covout) {

  // time between states
  double delT = dt*delTmult;

  // Smoothing factor diagonal
  // TODO: Make this parameter an input
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

  // convert input lists to vectors
  std::vector<np::ndarray> p2_vec = to_std_vector<np::ndarray>(p2_list);
  std::vector<np::ndarray> p1_vec = to_std_vector<np::ndarray>(p1_list);
  std::vector<np::ndarray> weight_vec = to_std_vector<np::ndarray>(weight_list);

  // uesful variables
  int window_size = poses.shape(0);

  //
  // Setup initial conditions
  //

  // Steam state variables
  std::vector<TrajStateVar> states;

  // Setup state variables
  for (unsigned int i = 0; i < window_size; i++) {
    Eigen::Matrix4d pose_mat = Eigen::Matrix4d::Identity();
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 4; ++c)
        pose_mat(r,c) = double(p::extract<float>(poses[i][0][r][c]));
    // std::cout << pose_mat << std::endl << std::endl;
    lgmath::se3::Transformation pose_lg(pose_mat);

    Eigen::Matrix<double,6,1> vel_vec;
    for (int r = 0; r < 6; ++r)
      vel_vec(r) = double(p::extract<float>(vels[i][r]));
    // std::cout << vel_vec.transpose() << std::endl << std::endl;

    TrajStateVar temp;
    temp.time = steam::Time(i*delT);
    temp.pose = steam::se3::TransformStateVar::Ptr(new steam::se3::TransformStateVar(pose_lg));
    temp.velocity = steam::VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(vel_vec));
    states.push_back(temp);
  } // end i

  // Setup Trajectory
  steam::se3::SteamTrajInterface traj(Qc_inv);
  for (unsigned int i = 0; i < states.size(); i++) {
    TrajStateVar& state = states.at(i);
    steam::se3::TransformStateEvaluator::Ptr temp =
        steam::se3::TransformStateEvaluator::MakeShared(state.pose);
    traj.add(state.time, temp, state.velocity);
    if (i == 0) // lock first pose
      state.pose->setLock(true);
  } // end i


  ///
  /// Setup Cost Terms
  ///

  steam::ParallelizedCostTermCollection::Ptr costTerms(new steam::ParallelizedCostTermCollection());
  traj.appendPriorCostTerms(costTerms);

  // steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());
  steam::GemanMcClureLossFunc::Ptr sharedLossFunc(new steam::GemanMcClureLossFunc(1.0));

  // loop through every frame
  for (unsigned int i = 1; i < window_size; i++) {
    // get pose
    auto T_k0_eval_ptr = traj.getInterpPoseEval(steam::Time(i*delT));

    int num_meas = p2_vec[i-1].shape(0);
    for (unsigned int j = 0; j < num_meas; ++j) {

      // get R
      Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
          R(r,c) = p::extract<double>(weight_vec[i-1][j][r][c]);
//      R += 1e-6*Eigen::Matrix3d::Identity();

      steam::BaseNoiseModel<3>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<3>(R, steam::INFORMATION));

      // get measurement
      Eigen::Vector4d read;
      read << double(p::extract<float>(p2_vec[i-1][j][0])), double(p::extract<float>(p2_vec[i-1][j][1])),
              double(p::extract<float>(p2_vec[i-1][j][2])), 1.0;

      Eigen::Vector4d ref;
      ref << double(p::extract<float>(p1_vec[i-1][j][0])), double(p::extract<float>(p1_vec[i-1][j][1])),
             double(p::extract<float>(p1_vec[i-1][j][2])), 1.0;

      steam::P2P3ErrorEval::Ptr error(new steam::P2P3ErrorEval(ref, read, T_k0_eval_ptr));
      steam::WeightedLeastSqCostTerm<3,6>::Ptr cost(
          new steam::WeightedLeastSqCostTerm<3,6>(error, sharedNoiseModel, sharedLossFunc));
      costTerms->add(cost);
    } // end j
  } // end i

  // Initialize problem
  steam::OptimizationProblem problem;

  // Add state variables
  for (unsigned int i = 0; i < states.size(); i++) {
    const TrajStateVar& state = states.at(i);
    if (i > 0)
      problem.addStateVariable(state.pose);
    problem.addStateVariable(state.velocity);
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

  // mean poses and velocities
  for (unsigned int i = 1; i < states.size(); i++) {
    const TrajStateVar& state = states.at(i);
    Eigen::Matrix4d T_21_eigen = state.pose->getValue().matrix();
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 4; ++c)
        poses[i][0][r][c] = float(T_21_eigen(r, c));

    Eigen::Matrix<double,6,1> vel_eigen = state.velocity->getValue();
    for (int r = 0; r < 6; ++r)
      vels[i][r] = float(vel_eigen(r));
  }

  if (!compute_sp) {
    // compute covariance between last two poses
    Eigen::MatrixXd cov;
    if (states.size() == 2){
      // no computation, just grab covariance of pose
      cov = solver.queryCovariance(states.back().pose->getKey());
    }
    else {
      // need to compute between 2 uncertain poses
      std::vector<steam::StateKey> keys;
      std::vector<lgmath::se3::Transformation> T_k0_mean;
      for (unsigned int i = states.size()-2; i < states.size(); i++) {
        const TrajStateVar& state = states.at(i);
        keys.push_back(state.pose->getKey());
        T_k0_mean.push_back(state.pose->getValue());
      }
      assert(keys.size() == 2);
      lgmath::se3::Transformation T_21_mean = T_k0_mean[1]*T_k0_mean[0].inverse();
      steam::BlockMatrix cov_blocks = solver.queryCovarianceBlock(keys);
      Eigen::Matrix<double,6,6> cov11 = cov_blocks.at(0, 0);
      Eigen::Matrix<double,6,6> cov12 = cov_blocks.at(0, 1);
      Eigen::Matrix<double,6,6> cov21 = cov_blocks.at(1, 0);
      Eigen::Matrix<double,6,6> cov22 = cov_blocks.at(1, 1);
      Eigen::Matrix<double,6,6> Tad_21 = lgmath::se3::tranAd(T_21_mean.matrix());
      cov = cov22 - Tad_21*cov12 - cov21*Tad_21.transpose() + Tad_21*cov11*Tad_21.transpose();
    }

    // to numpy
    for (int r = 0; r < 6; ++r)
      for (int c = 0; c < 6; ++c)
        covout[r][c] = float(cov(r, c));

    return;
  }

  // get sigmapoints

  // query covariance at once
  std::vector<steam::StateKey> keys;
  keys.reserve(window_size - 1);
  for (unsigned int i = 1; i < states.size(); i++) {
    const TrajStateVar& state = states.at(i);
    keys.push_back(state.pose->getKey());
  }

  steam::BlockMatrix cov_blocks = solver.queryCovarianceBlock(keys);

  // loop through every frame
  for (unsigned int i = 1; i < window_size; i++) {
    // get pose
//    auto T_k0_eval_ptr = traj.getInterpPoseEval(steam::Time(i*delT));

    // mean pose
    const TrajStateVar& state = states.at(i);
    Eigen::Matrix4d T_i0_eigen = state.pose->getValue().matrix();

    // get cov and LLT decomp
    int pose_block = i - 1;
    Eigen::Matrix<double,6,6> cov = cov_blocks.at(pose_block, pose_block);
    Eigen::LLT<Eigen::MatrixXd> lltcov(cov);
    Eigen::MatrixXd L = lltcov.matrixL();

    // sp
    int n = 6;  // 6 pose
    double alpha = sqrt(double(n));
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
          poses[i][a+1][r][c] = float(T_sp(r, c));
          poses[i][a+1+n][r][c] = float(T_sp_inv(r, c));
        } // end c
      } // end r

    } // end for a
  } // end for i

  return;
}

// boost python
BOOST_PYTHON_MODULE(steampy)
{
    Py_Initialize();
    np::initialize();

    p::def("run_steam", run_steam);
}
