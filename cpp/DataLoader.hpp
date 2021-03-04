#pragma once
#include <vector>
#include <string>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace p = boost::python;
namespace np = boost::python::numpy;

#define CTS350 0
#define CIR204 1

class DataLoader {
public:
    DataLoader(const double radar_resolution_, const double cart_resolution_,
        const uint cart_pixel_width_, const uint navtech_version_) : radar_resolution(radar_resolution_),
        cart_resolution(cart_resolution_), cart_pixel_width(cart_pixel_width_),
        navtech_version(navtech_version_) {
        if (navtech_version == CIR204) {
            range_bins = 3360;
        }
        min_range = uint(2.35 / radar_resolution);
    }

    // timestamps: N x 1 , azimuths: N x 1, fft_data: N x R need to be sized correctly by the python user.
    void load_radar(const std::string path, np::ndarray& timestamps, np::ndarray& azimuths, np::ndarray& fft_data);

    // azimuths: N x 1, fft_data: N x R, cart: W x W need to be sized correctly by the python user.
    void polar_to_cartesian(const np::ndarray& azimuths, const np::ndarray& fft_data, np::ndarray& cart);

private:
    double radar_resolution = 0.0432;
    double cart_resolution = 0.2160;
    uint cart_pixel_width = 640;
    uint navtech_version = 0;  // 0: CTS350, 1: CIR204
    uint range_bins = 3768;
    uint num_azimuths = 400;
    uint min_range = 54;
    bool interpolate_crossover = true;
};

class releaseGIL{
public:
    inline releaseGIL(){
        save_state = PyEval_SaveThread();
    }

    inline ~releaseGIL(){
        PyEval_RestoreThread(save_state);
	save_state = NULL;
    }
private:
    PyThreadState *save_state;
};

// boost wrapper
BOOST_PYTHON_MODULE(DataLoader) {
    PyEval_InitThreads();
    Py_Initialize();
    np::initialize();
    p::class_<DataLoader>("DataLoader", p::init<const double, const double, const uint, const uint>())
        .def("load_radar", &DataLoader::load_radar)
        .def("polar_to_cartesian", &DataLoader::polar_to_cartesian);
}
