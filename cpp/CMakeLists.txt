cmake_minimum_required(VERSION 3.30)

set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(nbn LANGUAGES CXX)

find_package(Vulkan REQUIRED)

include(FetchContent)

FetchContent_Declare(
    vmahpp
    URL https://github.com/YaaZ/VulkanMemoryAllocator-Hpp/releases/download/v3.3.0%2B3/VulkanMemoryAllocator-Hpp-3.3.0.tar.gz
    URL_HASH SHA256=8c2a0573babe1f86f3241d12b4c9df40bb944649b7f4239c1c611797f8af6cd5
)
FetchContent_MakeAvailable(vmahpp)

add_library(nbn device.cpp descriptors.cpp allocator.cpp util.cpp)
target_link_libraries(nbn Vulkan::Vulkan VulkanMemoryAllocator-Hpp::VulkanMemoryAllocator-Hpp)
target_precompile_headers(nbn PRIVATE pch.h)

add_executable(nbn_compute examples/compute.cpp)
target_link_libraries(nbn_compute nbn)
target_precompile_headers(nbn_compute REUSE_FROM nbn)

# Use extra warnings when compiling with clang.
target_compile_options(nbn PRIVATE
$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
    -Wall -Wextra -Wpedantic -Wfatal-errors -fdiagnostics-color=always
    >
)
