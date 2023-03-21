cmake_minimum_required(VERSION 3.20)

project(simple_texture)

add_definitions(-std=c++17)
set(CMAKE_CXX_STANDARD 17)

add_executable(simple_texture simple_texture.cpp)

set_target_properties(simple_texture PROPERTIES LINK_FLAGS "-s USE_WEBGPU=1")
