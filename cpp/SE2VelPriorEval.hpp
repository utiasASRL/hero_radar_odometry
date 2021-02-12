#ifndef SE2_VEL_PRIOR_EVAL_HPP
#define SE2_VEL_PRIOR_EVAL_HPP

#include <vector>
#include <steam.hpp>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Prior on velocity state for SE2 estimation problems
///
//////////////////////////////////////////////////////////////////////////////////////////////
class SE2VelPriorEval : public ErrorEvaluator<4, 6>::type {
public:
    /// Convenience typedefs
    typedef boost::shared_ptr<SE2VelPriorEval> Ptr;
    typedef boost::shared_ptr<const SE2VelPriorEval> ConstPtr;

    explicit SE2VelPriorEval(const VectorSpaceStateVar::ConstPtr& vel_state);

    virtual bool isActive() const;

    virtual Eigen::Matrix<double, 4, 1> evaluate() const;

    virtual Eigen::Matrix<double, 4, 1> evaluate(const Eigen::Matrix<double, 4, 4>& lhs,
        std::vector<Jacobian<4, 6>>* jacs) const;

private:
    VectorSpaceStateVar::ConstPtr vel_state_;
    Eigen::Matrix<double, 4, 6> D_;
};

}  // namespace steam

#endif  // SE2_VEL_PRIOR_EVAL_HPP
