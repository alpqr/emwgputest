cmake_minimum_required(VERSION 3.20)

project(localfile)

add_definitions(-std=c++17)
set(CMAKE_CXX_STANDARD 17)

add_executable(localfile localfile.cpp)

set(MEM_FLAGS "-sINITIAL_MEMORY=512MB -sALLOW_MEMORY_GROWTH=0 -sEXPORTED_FUNCTIONS=_main,_malloc,_free -sEXPORTED_RUNTIME_METHODS=ccall")
set_target_properties(localfile PROPERTIES LINK_FLAGS "-s USE_WEBGPU=1 ${MEM_FLAGS}")

set(imgui_base ../3rdparty/DearImgui)
set(imgui_sources
    ${imgui_base}/imgui.cpp
    ${imgui_base}/imgui_draw.cpp
    ${imgui_base}/imgui_tables.cpp
    ${imgui_base}/imgui_widgets.cpp
    ${imgui_base}/imgui_demo.cpp
)
target_sources(localfile PRIVATE
    ${imgui_sources}
)
target_include_directories(localfile PRIVATE
    ${imgui_base}
)
