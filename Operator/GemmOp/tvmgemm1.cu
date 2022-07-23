#ifdef _WIN32
using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short
#define int64_t long long
#define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(256)
default_function_kernel0(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C) {
    float C_local[32];
    __shared__ float A_shared[4096];
    __shared__ float B_shared[2048];
    float A_shared_local[8];
    float B_shared_local[4];
    for (int x_c_init = 0; x_c_init < 8; ++x_c_init) {
        for (int y_c_init = 0; y_c_init < 4; ++y_c_init) {
            C_local[(((x_c_init * 4) + y_c_init))] = 0.000000e+00f;
        }
    }
    for (int k_outer = 0; k_outer < 64; ++k_outer) {
        __syncthreads();
        for (int ax0_inner = 0; ax0_inner < 8; ++ax0_inner) {
            for (int ax1_inner = 0; ax1_inner < 2; ++ax1_inner) {
                A_shared[(((((((int) threadIdx.y) * 256) + (ax0_inner * 32)) + (((int) threadIdx.x) * 2)) +
                           ax1_inner))] = A[((
                        (((((((int) blockIdx.y) * 262144) + (((int) threadIdx.y) * 16384)) + (ax0_inner * 2048)) +
                          (k_outer * 32)) + (((int) threadIdx.x) * 2)) + ax1_inner))];
            }
        }
        for (int ax0_inner1 = 0; ax0_inner1 < 2; ++ax0_inner1) {
            for (int ax1_inner1 = 0; ax1_inner1 < 4; ++ax1_inner1) {
                B_shared[(((((((int) threadIdx.y) * 128) + (ax0_inner1 * 64)) + (((int) threadIdx.x) * 4)) +
                           ax1_inner1))] = B[((
                        (((((k_outer * 65536) + (((int) threadIdx.y) * 4096)) + (ax0_inner1 * 2048)) +
                          (((int) blockIdx.x) * 64)) + (((int) threadIdx.x) * 4)) + ax1_inner1))];
            }
        }



        __syncthreads();
        for (int k_inner = 0; k_inner < 32; ++k_inner) {
            for (int ax0 = 0; ax0 < 8; ++ax0) {
                A_shared_local[(ax0)] = A_shared[((((((int) threadIdx.y) * 256) + (ax0 * 32)) + k_inner))];
            }
            for (int ax1 = 0; ax1 < 4; ++ax1) {
                B_shared_local[(ax1)] = B_shared[((((k_inner * 64) + (((int) threadIdx.x) * 4)) + ax1))];
            }
            for (int x_c = 0; x_c < 8; ++x_c) {
                for (int y_c = 0; y_c < 4; ++y_c) {
                    C_local[(((x_c * 4) + y_c))] = (C_local[(((x_c * 4) + y_c))] +
                                                    (A_shared_local[(x_c)] * B_shared_local[(y_c)]));
                }
            }
        }

    }





    for (int x_inner = 0; x_inner < 8; ++x_inner) {
        for (int y_inner = 0; y_inner < 4; ++y_inner) {
            C[(((((((((int) blockIdx.y) * 262144) + (((int) threadIdx.y) * 16384)) + (x_inner * 2048)) +
                  (((int) blockIdx.x) * 64)) + (((int) threadIdx.x) * 4)) + y_inner))] = C_local[(((x_inner * 4) +
                                                                                                   y_inner))];
        }
    }
}