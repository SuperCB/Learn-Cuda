#include <cstdio>

#include <cuda_runtime.h>

#include "cuda.h"

__global__ void reduce(int *g_idata, int *g_odata, unsigned int len) {
    int tid = threadIdx.x;

    int *i_data = g_idata + blockDim.x * blockIdx.x;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            i_data[tid] += i_data[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = i_data[0];
    }
}

int main() {
    int len = 1 << 28;
    int blocksize = 1024;
    dim3 block(blocksize, 1);
    dim3 grid((len - 1) / blocksize + 1, 1);
    int *idata_host = (int *) malloc(len * sizeof(int));


    int *odata_host = (int *) malloc(grid.x * sizeof(int));

    for (int i = 0; i < len; i++)
        idata_host[i] = rand() % 1000;

    int *idata_dev = NULL;
    int *odata_dev = NULL;
    cudaMalloc(&idata_dev, len * sizeof(int));

    cudaMalloc(&odata_dev, grid.x * sizeof(int));
    int cpu_sum = 0;
    for (int i = 0; i < len; i++)
        cpu_sum += idata_host[i];

    cudaMemcpy(idata_dev, idata_host, len * sizeof(int), cudaMemcpyHostToDevice);

    reduce<<<grid, blocksize>>>(idata_dev, odata_dev, 1024);
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    int gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += odata_host[i];

    printf("cpu %d,gpu %d\n", cpu_sum, gpu_sum);
    if (cpu_sum == gpu_sum) {
        printf("success");
    }

    return 0;
}