// vim: ts=4:sw=4:noexpandtab
//////////////////////////////////////////////////////////////////////////////////////////////
/// \file P2PLandmarkErrorEval.cpp
///
/// \author David Yoon, ASRL
/// \brief This evaluator was develop in the context of ICP (Iterative Closest Point) 
///        implementation.
//////////////////////////////////////////////////////////////////////////////////////////////

#include "P2PLandmarkErrorEval.hpp"

namespace steam {


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
P2PLandmarkErrorEval::P2PLandmarkErrorEval(
		const se3::LandmarkStateVar::Ptr& ref_a,
		const Eigen::Vector4d& read_b,
		const se3::TransformEvaluator::ConstPtr& T_b_a
		): eval_(se3::compose(T_b_a, ref_a)),
		   read_b_(read_b){
  D_ << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0;
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool P2PLandmarkErrorEval::isActive() const {
	return eval_->isActive();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the 4-d measurement error (ul vl ur vr)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Vector3d P2PLandmarkErrorEval::evaluate() const {

	return D_*(read_b_ - homogModel(eval_->evaluate()));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the 4-d measurement error (ul vl ur vr) and Jacobians
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Vector3d P2PLandmarkErrorEval::evaluate(const Eigen::Matrix3d& lhs, std::vector<Jacobian<3,6> >* jacs) const {

	// Check and initialize jacobian array
	if (jacs == NULL) {
		throw std::invalid_argument("Null pointer provided to return-input 'jacs' in evaluate");
	}
	jacs->clear();

	// Get evaluation tree
	EvalTreeHandle<Eigen::Vector4d> blkAutoTransform =
		eval_->getBlockAutomaticEvaluation();
  // const Eigen::Vector4d& pointInCamFrame = blkAutoTransform.getValue();

	// Get Jacobians
  Eigen::Matrix<double, 3, 4> newLhs = -lhs*D_*homogModelJacobian(eval_->evaluate());
	eval_->appendBlockAutomaticJacobians(newLhs, blkAutoTransform.getRoot(), jacs);

	// Return evaluation
	return D_*(read_b_ - homogModel(eval_->evaluate()));
}

Eigen::Vector4d P2PLandmarkErrorEval::homogModel(const Eigen::Vector4d& point) const {

  // Precompute values
  const double x = point[0];
  const double y = point[1];
  const double z = point[2];
  const double w = point[3];

  // Project point into camera coordinates
  Eigen::Vector4d projectedMeas;
  projectedMeas << x/w, y/w, z/w, w/w;
  return projectedMeas;
}

Eigen::Matrix4d P2PLandmarkErrorEval::homogModelJacobian(const Eigen::Vector4d& point) const {
  const double x = point[0];
  const double y = point[1];
  const double z = point[2];
  const double w = point[3];
  const double one_over_w = 1/w;
  const double one_over_w2 = one_over_w*one_over_w;
  Eigen::Matrix4d jac;
  jac << one_over_w, 0.0, 0.0, -x*one_over_w2,
         0.0, one_over_w, 0.0, -y*one_over_w2,
         0.0, 0.0, one_over_w, -z*one_over_w2,
         0.0, 0.0, 0.0, 0.0;
  return jac;
}


} // steam
