cmake_minimum_required(VERSION 3.20)

project(localfile2)

add_definitions(-std=c++17)
set(CMAKE_CXX_STANDARD 17)

add_executable(localfile2 localfile2.cpp)

set(PRELOAD "--preload-file ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/fonts/RobotoMono-Medium.ttf@RobotoMono-Medium.ttf")
set(MEM_FLAGS "-sINITIAL_MEMORY=512MB -sALLOW_MEMORY_GROWTH=0")
set(OTHER_FLAGS "-sEXPORTED_FUNCTIONS=_main,_malloc,_free -sEXPORTED_RUNTIME_METHODS=ccall")
set_target_properties(localfile2 PROPERTIES LINK_FLAGS "-s USE_WEBGPU=1 ${MEM_FLAGS} ${OTHER_FLAGS} ${PRELOAD}")

add_custom_command(
    TARGET localfile2
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/localfile2.data
            ${CMAKE_CURRENT_SOURCE_DIR}/localfile2.data
)

set(imgui_base ../3rdparty/DearImgui)
set(imgui_sources
    ${imgui_base}/imgui.cpp
    ${imgui_base}/imgui_draw.cpp
    ${imgui_base}/imgui_tables.cpp
    ${imgui_base}/imgui_widgets.cpp
    ${imgui_base}/imgui_demo.cpp
)

target_sources(localfile2 PRIVATE
    ${imgui_sources}
)

target_include_directories(localfile2 PRIVATE
    ${imgui_base}
    ../3rdparty/glm
)
