#pragma once
#include <vector>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <steam/steam.hpp>

namespace p = boost::python;
namespace np = boost::python::numpy;

// trajectory state struct
struct TrajStateVar {
    steam::Time time;
    steam::se3::TransformStateVar::Ptr pose;
    steam::VectorSpaceStateVar::Ptr velocity;
};

// convert python list to std::vector
template<typename T> inline std::vector<T> toStdVector(const p::object& iterable) {
    return std::vector<T>(p::stl_input_iterator<T>(iterable), p::stl_input_iterator<T>( ));
}

// Convert 2D numpy array to double eigen matrix
Eigen::MatrixXd numpyToEigen2D(const np::ndarray& np_in) {
    uint row_size = np_in.shape(0);
    uint col_size = np_in.shape(1);
    Eigen::MatrixXd eig_out(row_size, col_size);
    for (uint r = 0; r < row_size; ++r) {
        for (uint c = 0; c < col_size; ++c) {
            eig_out(r, c) = double(p::extract<float>(np_in[r][c]));
        }
    }
    return eig_out;
}
