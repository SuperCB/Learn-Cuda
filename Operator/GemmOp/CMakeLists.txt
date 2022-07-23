
add_executable(gemm1 gemm1.cu)
add_executable(gemm2 gemm2.cu)
add_executable(gemm3 tvmgemm1.cu)

target_link_libraries(gemm2 cublas)