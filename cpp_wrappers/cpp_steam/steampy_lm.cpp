#include <steam/steam.hpp>
#include <boost/python.hpp>
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

template< typename T >
inline
std::vector< T > to_std_vector( const p::object& iterable ) {
  return std::vector< T >( p::stl_input_iterator< T >( iterable ),
                           p::stl_input_iterator< T >( ) );
}

// (meas_list, match_list, pose_list, vel_list, lm_coords)
char const* run_steam_lm(const p::object& meas_list, const p::object& match_list, const p::object& weight_list, 
    np::ndarray& poses, np::ndarray& vels, np::ndarray& lm_coords) {

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

  // convert input lists to vectors
  std::vector<np::ndarray> meas_vec = to_std_vector<np::ndarray>(meas_list);
  std::vector<np::ndarray> match_vec = to_std_vector<np::ndarray>(match_list);
  std::vector<np::ndarray> weight_vec = to_std_vector<np::ndarray>(weight_list);

  // uesful variables
  int window_size = poses.shape(0);
  int num_lm = lm_coords.shape(0);

  //
  // Setup initial conditions
  //

  // Steam state variables
  std::vector<TrajStateVar> states;
  std::vector<steam::se3::LandmarkStateVar::Ptr> landmarks;

  // Setup state variables
  for (unsigned int i = 0; i < window_size; i++) {
    Eigen::Matrix4d pose_mat = Eigen::Matrix4d::Identity();
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 4; ++c)
        pose_mat(r,c) = double(p::extract<float>(poses[i][r][c]));
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

  // Setup landmarks
  landmarks.resize(num_lm);
  for (unsigned int i = 0; i < num_lm; i++) {
    Eigen::Vector3d temp_lm;
    temp_lm << double(p::extract<float>(lm_coords[i][0])), double(p::extract<float>(lm_coords[i][1])),
               double(p::extract<float>(lm_coords[i][2]));
    landmarks[i] = steam::se3::LandmarkStateVar::Ptr(new steam::se3::LandmarkStateVar(temp_lm));
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

//  std::vector<np::ndarray> meas_vec = to_std_vector<np::ndarray>(list);
//  for (int i = 0; i < meas_vec.size(); ++i) {
//    std::cout << meas_vec[i].shape(0) << ", " << meas_vec[i].shape(1) << std::endl;
//  }

  steam::ParallelizedCostTermCollection::Ptr costTerms(new steam::ParallelizedCostTermCollection());
  traj.appendPriorCostTerms(costTerms);

  // steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());
  steam::GemanMcClureLossFunc::Ptr sharedLossFunc(new steam::GemanMcClureLossFunc(1.0));

  // loop through every frame
  for (unsigned int i = 0; i < window_size; i++) {
    // get pose
    auto T_k0_eval_ptr = traj.getInterpPoseEval(steam::Time(i*delT));

    // auto meas = &meas_vec[i];
    // auto matchs = &match_vec[i];
    // auto weights = &weight_vec[i];
    int num_meas = meas_vec[i].shape(0);
    for (unsigned int j = 0; j < num_meas; ++j) {

      // check landmark id and skip if negative (means we reject due to too few observations)
      int lm_id = p::extract<long>(match_vec[i][j]);
      if (lm_id < 0)
        continue;

      // get R
      Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
          R(r,c) = double(p::extract<float>(weight_vec[i][j][r][c]));
          // R(r,c) = double(p::extract<float>(weights[j][r][c]));

      steam::BaseNoiseModel<3>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<3>(R, steam::INFORMATION));

      // get measurement
      Eigen::Vector4d read;
      read << double(p::extract<float>(meas_vec[i][j][0])), double(p::extract<float>(meas_vec[i][j][1])),
              double(p::extract<float>(meas_vec[i][j][2])), 1.0;

      steam::P2PLandmarkErrorEval::Ptr error(new steam::P2PLandmarkErrorEval(landmarks[lm_id], read, T_k0_eval_ptr));
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
  // TODO: sigmapoints

  // mean poses and velocities
  for (unsigned int i = 1; i < states.size(); i++) {
    const TrajStateVar& state = states.at(i);
    Eigen::Matrix4d T_21_eigen = state.pose->getValue().matrix();
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 4; ++c)
        poses[i][r][c] = float(T_21_eigen(r, c));

    Eigen::Matrix<double,6,1> vel_eigen = state.velocity->getValue();
    for (int r = 0; r < 6; ++r)
      vels[i][r] = float(vel_eigen(r));
  }

  // mean landmarks
  // Add landmark variables
  for (unsigned int i = 0; i < landmarks.size(); i++) {
    const steam::se3::LandmarkStateVar::Ptr& landmark = landmarks.at(i);
    Eigen::Vector3d mean_lm = (landmark->getValue()/landmark->getValue()(3)).head<3>();
    lm_coords[i][0] = float(mean_lm(0));
    lm_coords[i][1] = float(mean_lm(1));
    lm_coords[i][2] = float(mean_lm(2));
  }

  return "hello, world";
}


BOOST_PYTHON_MODULE(steampy_lm)
{
    Py_Initialize();
    np::initialize();

    p::def("run_steam_lm", run_steam_lm);
}
