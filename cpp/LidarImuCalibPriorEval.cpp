//////////////////////////////////////////////////////////////////////////////////////////////
/// \file LidarImuCalibPriorEval.cpp
///
/// \author David Yoon, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include "LidarImuCalibPriorEval.hpp"
namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
LidarImuCalibPriorEval::LidarImuCalibPriorEval(const double& z_offset,
    const se3::TransformEvaluator::ConstPtr& T_radar_lidar)
    : z_offset_(z_offset), T_radar_lidar_(T_radar_lidar) {
  r_rl_in_l_.reset(new se3::PositionEvaluator(T_radar_lidar));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool LidarImuCalibPriorEval::isActive() const {
  return T_radar_lidar_->isActive();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the measurement error
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,3,1> LidarImuCalibPriorEval::evaluate() const {

  Eigen::Matrix<double,6,1> xi = lgmath::se3::tran2vec(T_radar_lidar_->evaluate().matrix());
  Eigen::Matrix<double,3,1> r_rl_in_l = r_rl_in_l_->evaluate();
  Eigen::Matrix<double,3,1> error;
  error << z_offset_ - r_rl_in_l(2), -xi(3), -xi(4);
  return error;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the measurement error and relevant Jacobians
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,3,1> LidarImuCalibPriorEval::evaluate(
    const Eigen::Matrix<double,3,3>& lhs,
    std::vector<Jacobian<3,6>>* jacs) const {

  // Check and initialize jacobian array
  if (jacs == NULL) {
    throw std::invalid_argument("Null pointer provided to return-input 'jacs' in evaluate");
  }
  jacs->clear();

  // T_s_v
  if (T_radar_lidar_->isActive()) {
    // z-offset
    EvalTreeHandle<Eigen::Matrix<double,3,1>> blkAutoEvalPosOfTransform =
       r_rl_in_l_->getBlockAutomaticEvaluation();
    Eigen::MatrixXd jacp(3,3);
    jacp.setZero();
    jacp(0, 2) = -1;
    r_rl_in_l_->appendBlockAutomaticJacobians(lhs*jacp, blkAutoEvalPosOfTransform.getRoot(), jacs);

    // elevation and roll offsets
    EvalTreeHandle<lgmath::se3::Transformation> blkAutoTransform =
      T_radar_lidar_->getBlockAutomaticEvaluation();
    Eigen::Matrix<double,3,6> jac;
    jac.setZero();
    jac(1, 3) = -1;
    jac(2, 4) = -1;
    T_radar_lidar_->appendBlockAutomaticJacobians(lhs*jac, blkAutoTransform.getRoot(), jacs);
  }

  // Return error
  return evaluate();
}


} // steam