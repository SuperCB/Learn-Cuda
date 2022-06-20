
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;
#define IDX2C(i, j, ld) (((i) * (ld)) + (j))
#define TILE_SIZE 16

void printPlainMatrix(const float* matrix, const int H, const int W)
{
    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            std::cout << std::fixed << std::setw(8) << std::setprecision(4) << matrix[IDX2C(i, j, W)];
        }
        std::cout << std::endl;
    }
}


__global__ void operator_matmul_h(const float *input1, const float *input2,
                                  float *output, int height, int k, int width) {
    __shared__ float shared_input1[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_input2[TILE_SIZE][TILE_SIZE];



    int bx = blockIdx.y;
    int by = blockIdx.x;
    int tx = threadIdx.y;
    int ty = threadIdx.x;

    int row = bx * TILE_SIZE + tx;
    int col = by * TILE_SIZE + ty;
    float v = 0;

    for (int i = 0; i < (int) (ceil((float) k / TILE_SIZE)); i++) {
        if (i * TILE_SIZE + ty < k && row < height)
            shared_input1[tx][ty] = input1[row * k + i * TILE_SIZE + ty];
        else
            shared_input1[tx][ty] = 0;

        if (i * TILE_SIZE + tx < k && col < width)
            shared_input2[tx][ty] = input2[(i * TILE_SIZE + tx) * width + col];
        else
            shared_input2[tx][ty] = 0;
        __syncthreads();

        for (int j = 0; j < TILE_SIZE; j++)
            v += shared_input1[tx][j] * shared_input2[j][ty];
        __syncthreads();
    }

    if (row < height && col < width) output[row * width + col] = v;
}


int main(){


    // Prepare input matrices
    float *A, *B, *C;
    int M, N, K;
    float alpha, beta;

    M = 200;
    N = 200;
    K = 300;
    alpha = 1.f;
    beta = 0.f;


    cudaMallocManaged((void**)&A, sizeof(float) * M * K);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            A[IDX2C(i, j, K)] = i + 1;
        }
    }

    /*
      A:
      1.0000  1.0000  1.0000
      2.0000  2.0000  2.0000
    */


    cudaMallocManaged((void**)&B, sizeof(float) * K * N);
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            B[IDX2C(i, j, N)] = i + 1;
        }
    }

    cudaMallocManaged((void**)&C, sizeof(float) * M * N);
    for (int i = 0 ; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[IDX2C(i, j, N) ] = 1;
        }
    }

    dim3 dim_block(TILE_SIZE, TILE_SIZE);
    dim3 dim_grid(ceil((float) M/ TILE_SIZE), ceil((float) N / TILE_SIZE));
    operator_matmul_h<<<dim_grid, dim_block>>>(A, B, C,
                                               N, K,M);



    return 0;
}