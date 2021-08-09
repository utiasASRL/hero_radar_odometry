// vim: ts=4:sw=4:noexpandtab
//////////////////////////////////////////////////////////////////////////////////////////////
/// \file PointToPointErrorEval.hpp
///
/// \author Francois Pomerleau, ASRL
/// \brief This evaluator was develop in the context of ICP (Iterative Closest Point)
///        implementation. Modified by David Yoon to change dimension from 4 to 2.
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STEAM_P2P2_ERROR_EVALUATOR_HPP
#define STEAM_P2P2_ERROR_EVALUATOR_HPP

#include <vector>
#include <steam.hpp>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief The distance between two points living in their respective frame is used as our
///        error function.
///
//////////////////////////////////////////////////////////////////////////////////////////////
class P2P2ErrorEval : public ErrorEvaluator<2, 6>::type {
public:
    /// Convenience typedefs
    typedef boost::shared_ptr<P2P2ErrorEval> Ptr;
    typedef boost::shared_ptr<const P2P2ErrorEval> ConstPtr;

    //////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Constructor
    /// \param ref_a     A point from the reference point cloud (static) expressed in homogeneous
    ///                  coordinates (i.e., [x, y, z, 1]) and in the frame a.
    /// \param T_a_world Transformation matrix from frame world to frame a.
    /// \param read_b    A point from the reading point cloud expressed in homogeneous
    ///                  coordinates (i.e., [x, y, z, 1]) and in the frame b.
    /// \param T_b_world Transformation matrix from frame world to frame b.
    //////////////////////////////////////////////////////////////////////////////////////////////
    P2P2ErrorEval(const Eigen::Vector4d& ref_a,
                  const se3::TransformEvaluator::ConstPtr& T_a_world,
                  const Eigen::Vector4d& read_b,
                  const se3::TransformEvaluator::ConstPtr& T_b_world);

    //////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Constructor
    /// \param ref_a     A point from the reference point cloud (static) expressed in homogeneous
    ///                  coordinates (i.e., [x, y, z, 1]) and in the frame a.
    /// \param read_b    A point from the reading point cloud expressed in homogeneous
    ///                  coordinates (i.e., [x, y, z, 1]) and in the frame b.
    /// \param T_a_b     Transformation matrix from frame b to frame a.
    //////////////////////////////////////////////////////////////////////////////////////////////
    P2P2ErrorEval(const Eigen::Vector4d& ref_a,
                  const Eigen::Vector4d& read_b,
                  const se3::TransformEvaluator::ConstPtr& T_b_a);

    //////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Returns whether or not an evaluator contains unlocked state variables
    //////////////////////////////////////////////////////////////////////////////////////////////
    virtual bool isActive() const;

    //////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Evaluate the 3-d measurement error (x, y, z)
    //////////////////////////////////////////////////////////////////////////////////////////////
    virtual Eigen::Vector2d evaluate() const;

    //////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Evaluate the 4-d measurement error (x, y, z) and Jacobians
    //////////////////////////////////////////////////////////////////////////////////////////////
    virtual Eigen::Vector2d evaluate(const Eigen::Matrix2d& lhs,
    std::vector<Jacobian<2, 6>>* jacs) const;

private:
    //////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Point evaluator (evaluates the point transformed into the camera frame)
    //////////////////////////////////////////////////////////////////////////////////////////////
    Eigen::Vector4d ref_a_;
    se3::TransformEvaluator::ConstPtr T_b_a_;
    Eigen::Vector4d read_b_;
    Eigen::Matrix<double, 2, 4> D_;
};

}  // namespace steam

#endif  // STEAM_P2P2_ERROR_EVALUATOR_HPP
