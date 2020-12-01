cmake_minimum_required(VERSION 3.10.2)
project(steampy)

# Activate c++17
execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(DEPEND_LIBS "boost_system")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Wno-unused-function")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# packages
find_package(PythonInterp)
# FIND_PACKAGE(Boost COMPONENTS python${PYTHON_VERSION_SUFFIX} numpy${PYTHON_VERSION_SUFFIX} system)
find_package(Boost COMPONENTS python36 numpy36 system)
find_package(PythonInterp 3)
find_package(PythonLibs 3 REQUIRED)

# openmp
find_package(OpenMP REQUIRED)

# steam
find_package(Steam REQUIRED)

# boost python
ENABLE_TESTING()
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${Steam_INCLUDE_DIRS})

PYTHON_ADD_MODULE(steampy steampy.cpp P2P3ErrorEval.cpp P2PLandmarkErrorEval.cpp)
target_link_libraries(steampy PRIVATE ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${Steam_LIBRARIES})