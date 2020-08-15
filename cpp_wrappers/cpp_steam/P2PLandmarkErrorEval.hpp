// vim: ts=4:sw=4:noexpandtab
//////////////////////////////////////////////////////////////////////////////////////////////
/// \file P2PLandmarkErrorEval.hpp
///
/// \author David Yoon, ASRL
/// \brief This evaluator was develop in the context of ICP (Iterative Closest Point) 
///        implementation. Modified by David Yoon to change dimension from 4 to 3.
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STEAM_P2P_LANDMARK_ERROR_EVALUATOR_HPP
#define STEAM_P2P_LANDMARK_ERROR_EVALUATOR_HPP

#include <steam.hpp>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief The distance between two points living in their respective frame is used as our
///        error function.
///
//////////////////////////////////////////////////////////////////////////////////////////////
class P2PLandmarkErrorEval : public ErrorEvaluator<3,6>::type
{
public:

	/// Convenience typedefs
	typedef boost::shared_ptr<P2PLandmarkErrorEval> Ptr;
	typedef boost::shared_ptr<const P2PLandmarkErrorEval> ConstPtr;

	//////////////////////////////////////////////////////////////////////////////////////////////
	/// \brief Constructor
	/// \param ref_a     A point from the reference point cloud (static) expressed in homogeneous 
	///                  coordinates (i.e., [x, y, z, 1]) and in the frame a.
	/// \param read_b    A point from the reading point cloud expressed in homogeneous 
	///                  coordinates (i.e., [x, y, z, 1]) and in the frame b.
	/// \param T_a_b     Transformation matrix from frame b to frame a.
	//////////////////////////////////////////////////////////////////////////////////////////////
	P2PLandmarkErrorEval(const se3::LandmarkStateVar::Ptr& ref_a,
						  const Eigen::Vector4d& read_b,
						  const se3::TransformEvaluator::ConstPtr& T_b_a
						  );

	//////////////////////////////////////////////////////////////////////////////////////////////
	/// \brief Returns whether or not an evaluator contains unlocked state variables
	//////////////////////////////////////////////////////////////////////////////////////////////
	virtual bool isActive() const; //TODO: check if we need to define that

	//////////////////////////////////////////////////////////////////////////////////////////////
	/// \brief Evaluate the 3-d measurement error (x, y, z)
	//////////////////////////////////////////////////////////////////////////////////////////////
	virtual Eigen::Vector3d evaluate() const;

	//////////////////////////////////////////////////////////////////////////////////////////////
	/// \brief Evaluate the 4-d measurement error (x, y, z) and Jacobians
	//////////////////////////////////////////////////////////////////////////////////////////////
	virtual Eigen::Vector3d evaluate(const Eigen::Matrix3d& lhs,
			std::vector<Jacobian<3,6> >* jacs) const;

  Eigen::Matrix4d homogModelJacobian(const Eigen::Vector4d& point) const;

private:
  Eigen::Vector4d homogModel(const Eigen::Vector4d& point) const;

	//////////////////////////////////////////////////////////////////////////////////////////////
	/// \brief Point evaluator (evaluates the point transformed into the camera frame)
	//////////////////////////////////////////////////////////////////////////////////////////////
	se3::ComposeLandmarkEvaluator::ConstPtr eval_;
	Eigen::Vector4d read_b_;
  Eigen::Matrix<double, 3, 4> D_;

};

} // steam

#endif // STEAM_P2P_LANDMARK_ERROR_EVALUATOR_HPP
