#include <cstdio>

#include <cuda_runtime.h>

#include "cuda.h"

#define THREAD_PER_BLOCK 256

__global__ void reduce1(float *g_idata, float *g_odata, unsigned int len) {


    __shared__ float sdata[THREAD_PER_BLOCK];
    int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i > len)
        return;

    sdata[tid] = g_idata[i];
    __syncthreads();
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce2(float *g_idata, float *g_odata, unsigned int len) {


    __shared__ float sdata[THREAD_PER_BLOCK];
    int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i > len)
        return;

    sdata[tid] = g_idata[i];
    __syncthreads();
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int loc = 2 * stride * tid;
        if (loc < blockDim.x) {
            sdata[loc] += sdata[loc + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce3(float *g_idata, float *g_odata, unsigned int len) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i > len)
        return;
    sdata[tid] = g_idata[i];
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
        if (tid < blockDim.x / 2) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}


__device__ void warpReduce(volatile float *cache, int tid) {

// can gpu reall
    cache[tid] += cache[tid + 32];
    cache[tid] += cache[tid + 16];
    cache[tid] += cache[tid + 8];
    cache[tid] += cache[tid + 4];
    cache[tid] += cache[tid + 2];
    cache[tid] += cache[tid + 1];
}
__global__ void reduce4(float *g_idata, float *g_odata, unsigned int len) {
    // re
    __shared__ float sdata[THREAD_PER_BLOCK];
    int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i > len)
        return;
    sdata[tid] = g_idata[i];
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < blockDim.x / 2) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid < 32)warpReduce(sdata, tid);//all thread in the same warp run the fuction at the same time
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__host__ void run_reduce4() {
    // let more thread can enroll in caculate
    int len = 1 << 30;
    int blocksize = 256;
    dim3 block(blocksize, 1);
    dim3 grid((len - 1) / blocksize + 1, 1);
    float *idata_host = (float *) malloc(len * sizeof(float));


    float *odata_host = (float *) malloc(grid.x * sizeof(float));

    for (int i = 0; i < len; i++)
        idata_host[i] = rand() % 1000;

    float *idata_dev = NULL;
    float *odata_dev = NULL;
    cudaMalloc(&idata_dev, len * sizeof(float));

    cudaMalloc(&odata_dev, grid.x * sizeof(float));
    float cpu_sum = 0.0;
    for (int i = 0; i < len; i++)
        cpu_sum += idata_host[i];
    printf("%f\n", cpu_sum);

    cudaMemcpy(idata_dev, idata_host, len * sizeof(float), cudaMemcpyHostToDevice);


    cudaEvent_t startTime, endTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&endTime);
    cudaEventRecord(startTime, 0);

    reduce1<<<grid, blocksize>>>(idata_dev, odata_dev, len);

    cudaEventRecord(endTime, 0);
    cudaEventSynchronize(startTime);
    cudaEventSynchronize(endTime);
    float time;
    cudaEventElapsedTime(&time, startTime, endTime);
    printf("time of reduce4 is %f ms\n", time);
    cudaDeviceSynchronize();
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    float gpu_sum = 0.0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += odata_host[i];

    printf("%f\n", gpu_sum);
}

int main() {
    int len = 1 << 30;
    int blocksize = 256;
    dim3 block(blocksize, 1);
    dim3 grid((len - 1) / blocksize + 1, 1);
    float *idata_host = (float *) malloc(len * sizeof(float));


    float *odata_host = (float *) malloc(grid.x * sizeof(float));

    for (int i = 0; i < len; i++)
        idata_host[i] = rand() % 1000;

    float *idata_dev = NULL;
    float *odata_dev = NULL;
    cudaMalloc(&idata_dev, len * sizeof(float));

    cudaMalloc(&odata_dev, grid.x * sizeof(float));
    float cpu_sum = 0.0;
    for (int i = 0; i < len; i++)
        cpu_sum += idata_host[i];
    printf("%f\n", cpu_sum);

    cudaMemcpy(idata_dev, idata_host, len * sizeof(float), cudaMemcpyHostToDevice);


    cudaEvent_t startTime, endTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&endTime);
    cudaEventRecord(startTime, 0);

    reduce4<<<grid, blocksize>>>(idata_dev, odata_dev, len);

    cudaEventRecord(endTime, 0);
    cudaEventSynchronize(startTime);
    cudaEventSynchronize(endTime);
    float time;
    cudaEventElapsedTime(&time, startTime, endTime);
    printf("time of reduce1 is %f ms\n", time);
    cudaDeviceSynchronize();


    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    float gpu_sum = 0.0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += odata_host[i];

    printf("%f\n", gpu_sum);

    cudaEventCreate(&startTime);
    cudaEventCreate(&endTime);
    cudaEventRecord(startTime, 0);

    reduce3<<<grid, blocksize>>>(idata_dev, odata_dev, len);

    cudaEventRecord(endTime, 0);
    cudaEventSynchronize(startTime);
    cudaEventSynchronize(endTime);
    cudaEventElapsedTime(&time, startTime, endTime);
    printf("time of reduce2 is %f ms\n", time);
    cudaDeviceSynchronize();


    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0.0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += odata_host[i];
    if (cpu_sum == gpu_sum) {
        printf("success");
    }

    return 0;
}