cmake_minimum_required(VERSION 2.8.3)
project(GraphOpt)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -march=native")
set(BUILD_EXAMPLES 1)

find_package(Eigen3 REQUIRED)
if (Eigen3_FOUND)
    message("EIGEN3 found.")
    message("EIGEN3_INCLUDE_DIRS: ${EIGEN3_INCLUDE_DIRS}")
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/gopt
    ${EIGEN3_INCLUDE_DIRS}
)

add_subdirectory(gopt)

if (${BUILD_EXAMPLES})
    message("Build examples.")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/example/curve_fitting)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/example/bundle_adjustment)
endif()
