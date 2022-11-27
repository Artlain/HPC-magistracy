#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>

using namespace std;
using namespace chrono;



#define SIZE 256
#define SHMEM_SIZE 256 * 4

system_clock::time_point device_start, device_end, host_start, host_end;

void sum_reduction_cpu(int* v, int n) {
	for (int i = 1; i < n; i++) {
		v[0] += v[i];
	}
}

__device__ void warpReduce(volatile int* shmem_ptr, int t) {
	shmem_ptr[t] += shmem_ptr[t + 32];
	shmem_ptr[t] += shmem_ptr[t + 16];
	shmem_ptr[t] += shmem_ptr[t + 8];
	shmem_ptr[t] += shmem_ptr[t + 4];
	shmem_ptr[t] += shmem_ptr[t + 2];
	shmem_ptr[t] += shmem_ptr[t + 1];
}

__global__ void sum_reduction(int* v, int* v_r) {
	__shared__ int partial_sum[SHMEM_SIZE];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
	__syncthreads();

	for (int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	if (threadIdx.x < 32) {
		warpReduce(partial_sum, threadIdx.x);
	}
	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = partial_sum[0];
	}
}

void initialize_vector(int* v, int n, int size) {
	for (int i = 0; i < n; i++) {
		v[i] = 1;
	}
	for (int i = n; i < size; i++) {
		v[i] = 0;
	}
	printf("Size %d \n", size);
	printf("Number of important elements %d \n", n);
}

int main() {
	int n = 1000000;

	int exponent = ceilf(log2f(n));
	int size = (int)powf(2, exponent);
	size_t bytes = size * sizeof(int);

	int* h_v, * h_v_r;
	int* d_v, * d_v_r;
	h_v = (int*)malloc(bytes);
	h_v_r = (int*)malloc(bytes);
	cudaMalloc(&d_v, bytes);
	cudaMalloc(&d_v_r, bytes);

	initialize_vector(h_v, n, size);
	cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);

	int TB_SIZE = 512;
	int GRID_SIZE = size / TB_SIZE / 2;

	device_start = system_clock::now();

	sum_reduction << <GRID_SIZE, TB_SIZE >> > (d_v, d_v_r);
	sum_reduction << <1, TB_SIZE >> > (d_v_r, d_v_r);

	device_end = system_clock::now();

	cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

	chrono::duration<float> device_duration = device_end - device_start;

	printf("Device result %d Device time %d\n", h_v_r[0], duration_cast<nanoseconds>(device_end - device_start).count());

	host_start = system_clock::now();
	sum_reduction_cpu(h_v, n);
	host_end = system_clock::now();
	printf("Host result %d Host time %d\n", h_v[0], duration_cast<nanoseconds>(host_end - host_start).count());
	return 0;
}
