cmake_minimum_required(VERSION 3.21)
project(learn_cuda )

set(CMAKE_CUDA_STANDARD 17)
find_package(CUDA REQUIRED)
cuda_add_executable(learn_cuda hello.cu)

#set_target_properties(learn_cuda PROPERTIES
