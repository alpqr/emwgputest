cmake_minimum_required(VERSION 3.20)

project(test1)

add_definitions(-std=c++17)
set(CMAKE_CXX_STANDARD 17)

add_executable(test1 test1.cpp)

#set_target_properties(test1 PROPERTIES LINK_FLAGS "-s USE_GLFW=3 -s USE_WEBGL2=1 -s FULL_ES3=1")
set_target_properties(test1 PROPERTIES LINK_FLAGS "-s USE_WEBGPU=1")
