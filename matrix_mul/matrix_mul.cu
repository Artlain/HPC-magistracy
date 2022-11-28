#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include<iostream>
#include<chrono>

using namespace std;
using namespace chrono;

system_clock::time_point device_start, device_end, host_start, host_end;

__global__ void matrixMul(int *a, int *b, int *c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N)
    {
        int tmp = 0;
        for (int i = 0; i < N; i++)
        {
            tmp += a[row * N + i] * b[i * N + col];
        }
        c[row * N + col] = tmp;
    }
}

void cpu_matrix_mul(int* a, int* b, int* h, int N) {
    int tmp;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            tmp = 0;
            for (int k = 0; k < N; k++) {
                tmp += a[i * N + k] * b[k * N + j];
            }
            h[i * N + j] = tmp;
        }
    }
}

bool check(int* c, int* l, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (c[i * N + j] != l[i * N + j]) {
                return false;
            }
        }
    }
    return true;
}

void matrix_initialization(int* m, int N) {
    for (int i = 0; i < N * N; i++) {
        m[i] = rand() % 100;
    }
}

int main()
{
    int N = 2000;
    size_t bytes = N * N * sizeof(int);

    int* a, * b, * c, *h;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);
    cudaMallocManaged(&h, bytes);

    matrix_initialization(a, N);
    matrix_initialization(b, N);
    int BLOCK_SIZE = 16;
    int GRID_SIZE = (N + BLOCK_SIZE - 1)/ BLOCK_SIZE;

    dim3 THREADS(BLOCK_SIZE, BLOCK_SIZE);
    dim3 BLOCKS(GRID_SIZE, GRID_SIZE);

    device_start = system_clock::now();
    matrixMul <<<BLOCKS, THREADS>>> (a, b, c, N);
    cudaDeviceSynchronize();
    device_end = system_clock::now();

    host_start = system_clock::now();
    cpu_matrix_mul(a, b, h, N);
    host_end = system_clock::now();

    bool is_equal = check(c, h, N);
    printf("Device time %d\n", duration_cast<milliseconds>(device_end - device_start).count());
    printf("Host time %d\n", duration_cast<milliseconds>(host_end - host_start).count());

    printf("Matrix is equal? %d\n", is_equal);
    return 0;
}
