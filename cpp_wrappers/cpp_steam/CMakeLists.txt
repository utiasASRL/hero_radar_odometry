cmake_minimum_required(VERSION 3.10.2)
project(example)

# Activate c++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# boost
# find_package(Boost REQUIRED COMPONENTS system)

# boost python
# set(Boost_NO_SYSTEM_PATHS ON)
# set(BOOST_ROOT "/home/david/anaconda3/envs/pointcloud")

FIND_PACKAGE(PythonInterp)
# FIND_PACKAGE(Boost COMPONENTS python${PYTHON_VERSION_SUFFIX} numpy${PYTHON_VERSION_SUFFIX} system)
FIND_PACKAGE(Boost COMPONENTS python36 numpy36 system)
FIND_PACKAGE(PythonInterp 3)
FIND_PACKAGE(PythonLibs 3 REQUIRED)

# openmp
find_package(OpenMP REQUIRED)

# steam
find_package(Steam REQUIRED)

# boost python
ENABLE_TESTING()
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${Steam_INCLUDE_DIRS})

# PYTHON_ADD_MODULE(hello hello.cpp)
# target_link_libraries(hello PRIVATE ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${Steam_LIBRARIES})

# PYTHON_ADD_MODULE(hello_np hello_np.cpp)
# target_link_libraries(hello_np PRIVATE ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${Steam_LIBRARIES})

PYTHON_ADD_MODULE(steampy_f2f steampy_f2f.cpp P2P3ErrorEval.cpp P2PLandmarkErrorEval.cpp)
target_link_libraries(steampy_f2f PRIVATE ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${Steam_LIBRARIES})

PYTHON_ADD_MODULE(steampy_lm steampy_lm.cpp P2P3ErrorEval.cpp P2PLandmarkErrorEval.cpp)
target_link_libraries(steampy_lm PRIVATE ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${Steam_LIBRARIES})

# regular
add_executable(landmark_eval_test landmark_eval_test.cpp P2PLandmarkErrorEval.cpp)
target_link_libraries(landmark_eval_test ${Boost_LIBRARIES} ${Steam_LIBRARIES} OpenMP::OpenMP_CXX)