cmake_minimum_required(VERSION 3.20)

project(blue_triangle)

add_definitions(-std=c++17)
set(CMAKE_CXX_STANDARD 17)

add_executable(blue_triangle blue_triangle.cpp)

set_target_properties(blue_triangle PROPERTIES LINK_FLAGS "-s USE_WEBGPU=1")
