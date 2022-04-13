
#include <cuda_runtime.h>
#include "cuda.h"
#include <cstdlib>
#include <cstdio>
#include <device_launch_parameters.h>
#define BLOCK_SIZE 16

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
//typedef struct {
//    int width;
//    int height;
//    float *elements;
//} Matrix;
//
//// Thread block size
//#define BLOCK_SIZE 16
//
//// Forward declaration of the matrix multiplication kernel
//__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
//
//// Matrix multiplication - Host code
//// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
//void MatMul(const Matrix A, const Matrix B, Matrix C) {
//    // Load A and B to device memory
//    Matrix d_A;
//    d_A.width = A.width;
//    d_A.height = A.height;
//    size_t size = A.width * A.height * sizeof(float);
//    cudaMalloc(&d_A.elements, size);
//    cudaMemcpy(d_A.elements, A.elements, size,
//               cudaMemcpyHostToDevice);
//    Matrix d_B;
//    d_B.width = B.width;
//    d_B.height = B.height;
//    size = B.width * B.height * sizeof(float);
//    cudaMalloc(&d_B.elements, size);
//    cudaMemcpy(d_B.elements, B.elements, size,
//               cudaMemcpyHostToDevice);
//
//    // Allocate C in device memory
//    Matrix d_C;
//    d_C.width = C.width;
//    d_C.height = C.height;
//    size = C.width * C.height * sizeof(float);
//    cudaMalloc(&d_C.elements, size);
//
//    // Invoke kernel
//    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
//    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
//    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
//
//    // Read C from device memory
//    cudaMemcpy(C.elements, C.elements, size,
//               cudaMemcpyDeviceToHost);
//
//    // Free device memory
//    cudaFree(d_A.elements);
//    cudaFree(d_B.elements);
//    cudaFree(d_C.elements);
//}
//
//// Matrix multiplication kernel called by MatMul()
//__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
//    // Each thread computes one element of C
//    // by accumulating results into Cvalue
//    float Cvalue = 0;
//    int row = blockIdx.y * blockDim.y + threadIdx.y;
//    int col = blockIdx.x * blockDim.x + threadIdx.x;
//    for (int e = 0; e < A.width; ++e)
//        Cvalue += A.elements[row * A.width + e]
//                  * B.elements[e * B.width + col];
//    C.elements[row * C.width + col] = Cvalue;
//}


typedef struct {
    int width;
    int height;
    int stride;
    float *elements;
} Matrix;

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col) {
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value) {
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                + BLOCK_SIZE * col];
    return Asub;
}
// Thread block size

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C) {
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);


    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    float Cvalue = 0;
    int row = threadIdx.y;
    int col = threadIdx.x;
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        __syncthreads();
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        __syncthreads();
    }
    SetElement(Csub, row, col, Cvalue);
}

void cpuMul(Matrix *A,Matrix* B,Matrix*C )
{

}
int main() {

    int len = 1024;
    Matrix A, B, C;
    A.width = 1024;
    A.height = 1024;
    B.width = 1024;
    B.height = 1024;
    C.width = 1024;
    C.height = 1024;
    A.stride = 32;
    B.stride = 32;
    A.elements = new float[len * len];
    B.elements = new float[len * len];
    C.elements = new float[len * len];
    for (int i = 0; i < 1024; i++)
        for (int j = 0; j < 1024; j++) {
            A.elements[i * len + j] = B.elements[i * len + j] = C.elements[i * len + j] = (float) (rand()%100);
        }



    cudaEvent_t startTime, endTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&endTime);

    cudaEventRecord(startTime, 0);

    MatMul(A, B, C);

    cudaEventRecord(endTime, 0);
    cudaEventSynchronize(startTime);
    cudaEventSynchronize(endTime);

    float time;
    cudaEventElapsedTime(&time, startTime, endTime);
    printf("%f ms\n", time);
    for (int i = 0; i < 1024; i++)
        for (int j = 0; j < 1024; j++) {
            float t=0.0;
            for(int k=0;k<1024;k++)
            {
                t+=A.elements[i*len+k]*B.elements[k*len+j];
            }

                printf("%f %f\n",t,C.elements[i * len + j]);
        }


    return 0;
}