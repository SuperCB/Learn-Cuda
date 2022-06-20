cmake_minimum_required(VERSION 3.21)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.6/bin/nvcc)

project(learn_cuda LANGUAGES CXX CUDA)

# 上面是cuda的环境变量

#set(CMAKE_CUDA_STANDARD 17)
#FindCUDAToolkit(cuBLAS)
#add_executable(Dive-in-Cuda learn.cu)
#target_link_libraries(Dive-in-Cuda cublas)
add_executable(main main.cu)

#set_target_properties(Dive-in-Cuda PROPERTIES

add_subdirectory(learn_cublass)
add_subdirectory(Operator)