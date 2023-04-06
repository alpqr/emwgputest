cmake_minimum_required(VERSION 3.20)

project(textures)

add_definitions(-std=c++17)
set(CMAKE_CXX_STANDARD 17)

add_executable(textures textures.cpp)

set(PRELOAD "--preload-file ${CMAKE_CURRENT_SOURCE_DIR}/test.png@test.png --preload-file ${CMAKE_CURRENT_SOURCE_DIR}/OpenfootageNET_lowerAustria01-1024.exr@test.exr")
set(MEM_FLAGS "-sINITIAL_MEMORY=512MB -sALLOW_MEMORY_GROWTH=0")
set_target_properties(textures PROPERTIES LINK_FLAGS "-s USE_WEBGPU=1 ${MEM_FLAGS} ${PRELOAD}")

add_custom_command(
    TARGET textures
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/textures.data
            ${CMAKE_CURRENT_SOURCE_DIR}/textures.data
)
