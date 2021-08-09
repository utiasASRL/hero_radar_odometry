// vim: ts=4:sw=4:noexpandtab
//////////////////////////////////////////////////////////////////////////////////////////////
/// \file P2P2ErrorEval.cpp
///
/// \author Francois Pomerleau, ASRL
/// \brief This evaluator was develop in the context of ICP (Iterative Closest Point)
///        implementation.
//////////////////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include "P2P2ErrorEval.hpp"

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
P2P2ErrorEval::P2P2ErrorEval(
    const Eigen::Vector4d& ref_a,
    const se3::TransformEvaluator::ConstPtr& T_a_world,
    const Eigen::Vector4d& read_b,
    const se3::TransformEvaluator::ConstPtr& T_b_world):
    ref_a_(ref_a),
    T_b_a_(se3::ComposeInverseTransformEvaluator::MakeShared(T_a_world, T_b_world)),
    read_b_(read_b) {
    D_ << 1, 0, 0, 0,
          0, 1, 0, 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
P2P2ErrorEval::P2P2ErrorEval(
    const Eigen::Vector4d& ref_a,
    const Eigen::Vector4d& read_b,
    const se3::TransformEvaluator::ConstPtr& T_b_a):
    ref_a_(ref_a),
    T_b_a_(T_b_a),
    read_b_(read_b){
    D_ << 1, 0, 0, 0,
          0, 1, 0, 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool P2P2ErrorEval::isActive() const {
    return T_b_a_->isActive();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the 4-d measurement error (ul vl ur vr)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Vector2d P2P2ErrorEval::evaluate() const {
    // Return error (between measurement and point estimate projected in camera frame)
    return D_ * (read_b_ - T_b_a_->evaluate() * ref_a_);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the 4-d measurement error (ul vl ur vr) and Jacobians
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Vector2d P2P2ErrorEval::evaluate(const Eigen::Matrix2d& lhs, std::vector<Jacobian<2, 6>>* jacs) const {
    // Check and initialize jacobian array
    if (jacs == NULL) {
        throw std::invalid_argument("Null pointer provided to return-input 'jacs' in evaluate");
    }
    jacs->clear();
    // Get evaluation tree
    EvalTreeHandle<lgmath::se3::Transformation> blkAutoTransform = T_b_a_->getBlockAutomaticEvaluation();
    // Get evaluation from tree
    const lgmath::se3::Transformation T_ba = blkAutoTransform.getValue();
    const Eigen::Vector4d ref_b = T_ba * ref_a_;
    // Get Jacobians
    Eigen::Matrix<double, 2, 6> newLhs = -1 * lhs * D_ * lgmath::se3::point2fs(ref_b.head<3>());
    T_b_a_->appendBlockAutomaticJacobians(newLhs, blkAutoTransform.getRoot(), jacs);
    // Return evaluation
    // return ref_a_ - read_a;
    return D_ * (read_b_ - ref_b);
}

}  // namespace steam
