cmake_minimum_required(VERSION 2.8.3)
project(curve_fitting)

# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -march=native")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../gopt
    ${EIGEN3_INCLUDE_DIRS}
)

message("Add example ${PROJECT_NAME}")
add_executable(${PROJECT_NAME} curve_fitting.cpp)
target_link_libraries(${PROJECT_NAME} ${LIB_NAME}_util ${LIB_NAME}_solver ${LIB_NAME}_graph)
