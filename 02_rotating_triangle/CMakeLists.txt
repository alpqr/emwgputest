cmake_minimum_required(VERSION 3.20)

project(rotating_triangle)

add_definitions(-std=c++17)
set(CMAKE_CXX_STANDARD 17)

add_executable(rotating_triangle rotating_triangle.cpp)

set_target_properties(rotating_triangle PROPERTIES LINK_FLAGS "-s USE_WEBGPU=1")
