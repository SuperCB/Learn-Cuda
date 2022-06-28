
add_executable(gemm1 gemm1.cu)

target_link_libraries(gemm1 cublas)