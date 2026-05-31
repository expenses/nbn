cmake_minimum_required(VERSION 3.30)

set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(nbn LANGUAGES CXX)

find_package(Vulkan REQUIRED)

add_executable(nbn main.cpp device.cpp descriptors.cpp)
target_link_libraries(nbn Vulkan::Vulkan)
target_precompile_headers(nbn PRIVATE pch.h)

# Use extra warnings when compiling with clang.
target_compile_options(nbn PRIVATE
$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
    -Wall -Wextra -Wpedantic -Wfatal-errors -fdiagnostics-color=always -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-pre-c++20-compat-pedantic -Wno-extra-semi-stmt -Wno-missing-prototypes -Wno-double-promotion -Wno-switch-default
    >
)
