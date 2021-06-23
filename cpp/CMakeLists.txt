cmake_minimum_required(VERSION 3.10.2)
project(steampy)

# Activate c++17
execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 ${OpenMP_CXX_FLAGS}")
set(OMP_NUM_THREADS 8)

set(DEPEND_LIBS "boost_system")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Wno-unused-function")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# packages
find_package(PythonInterp)
find_package(Boost COMPONENTS
    python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}
    numpy${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}
    system
)
find_package(PythonInterp 3)
find_package(PythonLibs 3 REQUIRED)

# openmp
#find_package(OpenMP REQUIRED)
#set(OMP_NUM_THREADS 8)

# steam
find_package(Steam REQUIRED)

# opencv
find_package(OpenCV 3 REQUIRED)

# boost python
ENABLE_TESTING()
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${Steam_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

# single function call version
PYTHON_ADD_MODULE(steampy steampy.cpp P2P3ErrorEval.cpp)
target_link_libraries(steampy PRIVATE ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${Steam_LIBRARIES})

# class version
PYTHON_ADD_MODULE(SteamSolver SteamSolver.cpp P2P3ErrorEval.cpp SE2VelPriorEval.cpp mcransac.cpp)
target_link_libraries(SteamSolver PRIVATE ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${Steam_LIBRARIES})

# dataloader
PYTHON_ADD_MODULE(DataLoader DataLoader.cpp)
target_link_libraries(DataLoader PRIVATE ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${Steam_LIBRARIES} ${OpenCV_LIBRARIES})
