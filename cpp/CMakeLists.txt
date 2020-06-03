cmake_minimum_required(VERSION 3.10)

project(diff_pd)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions("-Wall")
set(GCC_COVERAGE_COMPILE_FLAGS "-Wl,--no-as-needed")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")

# Include directories.
include_directories("./lib/eigen-3.3.7")
include_directories("./core/")
execute_process(COMMAND bash -c "python3-config --includes|sed 's/-I//g'|awk '{print $1;}'" OUTPUT_VARIABLE PYTHON_INCLUDES)
include_directories(${PYTHON_INCLUDES})

# The core library.
file(GLOB_RECURSE CORE_CPP "./core/*.cpp")
add_library(diff_pd_core SHARED ${CORE_CPP})

# Python wrapper library.
file(GLOB_RECURSE PYTHON_BINDING "./core/*.cpp" "./core/py_diff_pd_wrap.cxx")
add_library(py_diff_pd SHARED ${CORE_HEADER} ${PYTHON_BINDING})

# Demo.
file(GLOB DEMO_CPP "./main.cpp")
add_executable(diff_pd_demo ${DEMO_CPP})
target_link_libraries(diff_pd_demo diff_pd_core)