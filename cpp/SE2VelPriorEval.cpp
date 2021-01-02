#include "SE2VelPriorEval.hpp"

namespace steam {


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
SE2VelPriorEval::SE2VelPriorEval(
    const VectorSpaceStateVar::ConstPtr& vel_state):
        vel_state_(vel_state) {
  D_ << 0, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool SE2VelPriorEval::isActive() const {
  return (!vel_state_->isLocked());
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the error
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,3,1> SE2VelPriorEval::evaluate() const {
  return D_*vel_state_->getValue();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the error and Jacobians
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,3,1> SE2VelPriorEval::evaluate(const Eigen::Matrix<double,3,3>& lhs,
    std::vector<Jacobian<3,6> >* jacs) const {

  // Check and initialize jacobian array
  if (jacs == NULL) {
    throw std::invalid_argument("Null pointer provided to return-input 'jacs' in evaluate");
  }
  jacs->clear();

  // Construct Jacobian
  if(!vel_state_->isLocked()) {
    jacs->push_back(Jacobian<3,6>());
    Jacobian<3,6>& jacref = jacs->back();
    jacref.key = vel_state_->getKey();
    jacref.jac = lhs*D_;
  }

  return D_*vel_state_->getValue();
}


} // steam