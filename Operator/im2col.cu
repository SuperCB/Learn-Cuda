//
// Created by supercb on 22-6-28.
//
#define BLOCK_SIZE 256
__global__ void im2col_h(const int n, const float *data_im, const int height,
                         const int width, const int kernel_h,
                         const int kernel_w, const int pad_h, const int pad_w,
                         const int stride_h, const int stride_w,
                         const int height_col, const int width_col,
                         float *data_col, int im_stride, int col_stride) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        const int batch_idx = blockIdx.y;
        data_im += batch_idx * im_stride;
        data_col += batch_idx * col_stride;

        const int h_index = index / width_col;
        const int h_col = h_index % height_col;
        const int w_col = index % width_col;
        const int c_im = h_index / height_col;
        const int c_col = c_im * kernel_h * kernel_w;
        const int h_offset = h_col * stride_h - pad_h;
        const int w_offset = w_col * stride_w - pad_w;

        // channel offset
        float *data_col_ptr = data_col;
        data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
        const float *data_im_ptr = data_im;
        data_im_ptr += (c_im * height + h_offset) * width + w_offset;

        // copy to col
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                int h_im = h_offset + i;
                int w_im = w_offset + j;
                *data_col_ptr =
                        (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                        ? data_im_ptr[i * width + j]
                        : 0;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

void im2col(const float *data_im, const int batch_size, const int channels,
            const int height, const int width, const int kernel_h,
            const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, float *data_col) {
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int size = channels * height_col * width_col;

    int im_stride = channels * height * width;
    int col_stride = channels * kernel_h * kernel_w * height_col * width_col;
    dim3 dim_grid(ceil((float)size / BLOCK_SIZE), batch_size);
    im2col_h<<<dim_grid, BLOCK_SIZE>>>(
            size, data_im, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h,
            stride_w, height_col, width_col, data_col, im_stride, col_stride);
    CUDA_POST_KERNEL_CHECK;
}

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
extern "C" __global__ void __launch_bounds__(256) default_function_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
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
                A_shared[(((((((int)threadIdx.y) * 256) + (ax0_inner * 32)) + (((int)threadIdx.x) * 2)) + ax1_inner))] = A[(((((((((int)blockIdx.y) * 262144) + (((int)threadIdx.y) * 16384)) + (ax0_inner * 2048)) + (k_outer * 32)) + (((int)threadIdx.x) * 2)) + ax1_inner))];
            }
        }
        for (int ax0_inner1 = 0; ax0_inner1 < 2; ++ax0_inner1) {
            for (int ax1_inner1 = 0; ax1_inner1 < 4; ++ax1_inner1) {
                B_shared[(((((((int)threadIdx.y) * 128) + (ax0_inner1 * 64)) + (((int)threadIdx.x) * 4)) + ax1_inner1))] = B[(((((((k_outer * 65536) + (((int)threadIdx.y) * 4096)) + (ax0_inner1 * 2048)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 4)) + ax1_inner1))];
            }
        }
        __syncthreads();
        for (int k_inner = 0; k_inner < 32; ++k_inner) {
            for (int ax0 = 0; ax0 < 8; ++ax0) {
                A_shared_local[(ax0)] = A_shared[((((((int)threadIdx.y) * 256) + (ax0 * 32)) + k_inner))];
            }
            for (int ax1 = 0; ax1 < 4; ++ax1) {
                B_shared_local[(ax1)] = B_shared[((((k_inner * 64) + (((int)threadIdx.x) * 4)) + ax1))];
            }
            for (int x_c = 0; x_c < 8; ++x_c) {
                for (int y_c = 0; y_c < 4; ++y_c) {
                    C_local[(((x_c * 4) + y_c))] = (C_local[(((x_c * 4) + y_c))] + (A_shared_local[(x_c)] * B_shared_local[(y_c)]));
                }
            }
        }
    }
    for (int x_inner = 0; x_inner < 8; ++x_inner) {
        for (int y_inner = 0; y_inner < 4; ++y_inner) {
            C[(((((((((int)blockIdx.y) * 262144) + (((int)threadIdx.y) * 16384)) + (x_inner * 2048)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 4)) + y_inner))] = C_local[(((x_inner * 4) + y_inner))];
        }
    }
}