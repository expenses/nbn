cmake_minimum_required(VERSION 3.30)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(nbn LANGUAGES CXX)

find_package(Vulkan REQUIRED)

add_executable(nbn main.cpp device.cpp)
target_link_libraries(nbn Vulkan::Vulkan)
target_precompile_headers(nbn PRIVATE pch.h)
