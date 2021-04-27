#include "P2P3ErrorEval.hpp"
#include "SteamPyHelper.hpp"

namespace p = boost::python;
namespace np = boost::python::numpy;

// run steam optimization
void run_steam(const p::object& p2_list, const p::object& p1_list, const p::object& weight_list,
    np::ndarray& poses, np::ndarray& vels, bool compute_sp, double dt) {

    // time between states
    double delT = dt;

    // Smoothing factor diagonal
    Eigen::Array<double, 1, 6> Qc_diag;
    Qc_diag << 0.3678912639416186958207788393338,
               0.043068034591947058908889545136844,
               0.1307444996557916849777569723301,
               0.0073124100132336252236275875304727,
               0.0076438703775169331705585662461999,
               0.0021394075786459413462958778495704;

    Eigen::Matrix<double, 6, 6> Qc_inv; Qc_inv.setZero();
    Qc_inv.diagonal() = 1.0/Qc_diag;

    // convert input lists to vectors
    std::vector<np::ndarray> p2_vec = toStdVector<np::ndarray>(p2_list);
    std::vector<np::ndarray> p1_vec = toStdVector<np::ndarray>(p1_list);
    std::vector<np::ndarray> weight_vec = toStdVector<np::ndarray>(weight_list);

    // useful variables
    int window_size = poses.shape(0);

    //
    // Setup initial conditions
    //

    // Steam state variables
    std::vector<TrajStateVar> states;

    // Setup state variables
    for (uint i = 0; i < window_size; ++i) {
        Eigen::Matrix4d pose_mat = Eigen::Matrix4d::Identity();
        for (uint r = 0; r < 3; ++r) {
            for (uint c = 0; c < 4; ++c) {
                pose_mat(r, c) = double(p::extract<float>(poses[i][0][r][c]));
            }
        }
        lgmath::se3::Transformation pose_lg(pose_mat);

        Eigen::Matrix<double, 6, 1> vel_vec;
        for (uint r = 0; r < 6; ++r) {
            vel_vec(r) = double(p::extract<float>(vels[i][r]));
        }
        TrajStateVar temp;
        temp.time = steam::Time(i*delT);
        temp.pose = steam::se3::TransformStateVar::Ptr(new steam::se3::TransformStateVar(pose_lg));
        temp.velocity = steam::VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(vel_vec));
        states.push_back(temp);
    }  // end i

    // Setup Trajectory
    steam::se3::SteamTrajInterface traj(Qc_inv);
    for (uint i = 0; i < states.size(); ++i) {
        TrajStateVar& state = states.at(i);
        steam::se3::TransformStateEvaluator::Ptr temp = steam::se3::TransformStateEvaluator::MakeShared(state.pose);
        traj.add(state.time, temp, state.velocity);
        if (i == 0)  // lock first pose
            state.pose->setLock(true);
    }  // end i

    ///
    /// Setup Cost Terms
    ///
    steam::ParallelizedCostTermCollection::Ptr costTerms(new steam::ParallelizedCostTermCollection());
    traj.appendPriorCostTerms(costTerms);

    // steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());
    steam::GemanMcClureLossFunc::Ptr sharedLossFunc(new steam::GemanMcClureLossFunc(1.0));

    // loop through every frame
    for (uint i = 1; i < window_size; ++i) {
        auto T_k0_eval_ptr = traj.getInterpPoseEval(steam::Time(i * delT));

        int num_meas = p2_vec[i-1].shape(0);
        for (uint j = 0; j < num_meas; ++j) {
            Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
            for (uint r = 0; r < 3; ++r) {
                for (uint c = 0; c < 3; ++c) {
                    R(r, c) = p::extract<float>(weight_vec[i - 1][j][r][c]);
                    // R += 1e-6*Eigen::Matrix3d::Identity();
                }
            }
            steam::BaseNoiseModel<3>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<3>(R, steam::INFORMATION));

            // get measurement
            Eigen::Vector4d read;
            read << double(p::extract<float>(p2_vec[i-1][j][0])), double(p::extract<float>(p2_vec[i-1][j][1])),
                  double(p::extract<float>(p2_vec[i-1][j][2])), 1.0;

            Eigen::Vector4d ref;
            ref << double(p::extract<float>(p1_vec[i-1][j][0])), double(p::extract<float>(p1_vec[i-1][j][1])),
                 double(p::extract<float>(p1_vec[i-1][j][2])), 1.0;

            steam::P2P3ErrorEval::Ptr error(new steam::P2P3ErrorEval(ref, read, T_k0_eval_ptr));
            steam::WeightedLeastSqCostTerm<3, 6>::Ptr cost(
                new steam::WeightedLeastSqCostTerm<3, 6>(error, sharedNoiseModel, sharedLossFunc));
            costTerms->add(cost);
        }  // end j
    }  // end i

    // Initialize problem
    steam::OptimizationProblem problem;

    // Add state variables
    for (uint i = 0; i < states.size(); ++i) {
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
    for (uint i = 1; i < states.size(); ++i) {
        const TrajStateVar& state = states.at(i);
        Eigen::Matrix4d T_21_eigen = state.pose->getValue().matrix();
        for (uint r = 0; r < 3; ++r) {
            for (uint c = 0; c < 4; ++c) {
                poses[i][0][r][c] = float(T_21_eigen(r, c));
            }
        }
        Eigen::Matrix<double, 6, 1> vel_eigen = state.velocity->getValue();
        for (uint r = 0; r < 6; ++r) {
            vels[i][r] = float(vel_eigen(r));
        }
    }  // end i

    if (!compute_sp)
        return;

    // get sigmapoints
    // query covariance at once
    std::vector<steam::StateKey> keys;
    keys.reserve(window_size - 1);
    for (uint i = 1; i < states.size(); ++i) {
        const TrajStateVar& state = states.at(i);
        keys.push_back(state.pose->getKey());
    }

    steam::BlockMatrix cov_blocks = solver.queryCovarianceBlock(keys);

    // loop through every frame
    for (uint i = 1; i < window_size; ++i) {
        // get pose
        //    auto T_k0_eval_ptr = traj.getInterpPoseEval(steam::Time(i*delT));

        // mean pose
        const TrajStateVar& state = states.at(i);
        Eigen::Matrix4d T_i0_eigen = state.pose->getValue().matrix();

        // get cov and LLT decomp
        int pose_block = i - 1;
        Eigen::Matrix<double, 6, 6> cov = cov_blocks.at(pose_block, pose_block);
        Eigen::LLT<Eigen::MatrixXd> lltcov(cov);
        Eigen::MatrixXd L = lltcov.matrixL();

        // sp
        uint n = 6;  // 6 pose
        double alpha = sqrt(double(n));
        for (uint a = 0; a < n; ++a) {
            // delta for pose
            Eigen::Matrix4d T_sp = lgmath::se3::vec2tran(L.col(a).head<6>() * alpha);
            Eigen::Matrix4d T_sp_inv = lgmath::se3::vec2tran(-1 * L.col(a).head<6>() * alpha);

            // positive/negative sigmapoints
            T_sp = T_sp * T_i0_eigen;
            T_sp_inv = T_sp_inv * T_i0_eigen;

            // set output
            for (uint r = 0; r < 4; ++r) {
                for (uint c = 0; c < 4; ++c) {
                    poses[i][a + 1][r][c] = float(T_sp(r, c));
                    poses[i][a + 1 + n][r][c] = float(T_sp_inv(r, c));
                }  // end for c
            }  // end for r
        }  // end for a
    }  // end for i
    return;
}

// boost python
BOOST_PYTHON_MODULE(steampy)
{
    Py_Initialize();
    np::initialize();
    p::def("run_steam", run_steam);
}
