
set(CMAKE_CXX_STANDARD 20)

cmake_minimum_required(VERSION 3.18)

project(backprop LANGUAGES CXX CUDA)

add_subdirectory(src)
add_subdirectory(examples)