
/**
 * Copyright 1993-2012 Zerotech Corporation.  All rights reserved.
 *
 * work of wtd.
 *
 *
 */

#include "cuda_runtime.h"
#include "curand.h"
#include "cuda.h"

extern "C" {
#include "gemm.h"
}

#define BLOCK 512



__global__ void ztSgmvStrideBatched_3x3(const float * __restrict__ x_gpu, int K,
		const float * __restrict__ A_gpu, int N, int groups, float * __restrict__ C_gpu) {

	__shared__ float x[BLOCK];
	int thread_index_in_block = threadIdx.x;
	int total_index_in_group = threadIdx.x + blockDim.x * blockIdx.x;

	int kernel_group_offset = blockIdx.y * K;
	int input_data_group_offset = blockIdx.y * N * K;
	int output_data_group_offset = blockIdx.y * N;

	if(thread_index_in_block < K) {
		x[thread_index_in_block] = x_gpu[kernel_group_offset + thread_index_in_block];
	}

	__syncthreads();

	float acc = 0.0;

	if(total_index_in_group < N) {
#pragma unroll
		for(int i = 0; i < 9; i++) {
			acc += x[i] * A_gpu[input_data_group_offset + i*N + total_index_in_group];

		}
		C_gpu[output_data_group_offset + total_index_in_group] = acc;
	}

}

__global__ void ztSgmvStrideBatched_nxn(const float * __restrict__ x_gpu, int K,
		const float * __restrict__ A_gpu, int N, int groups, float * __restrict__ C_gpu) {

	__shared__ float x[BLOCK];
	int thread_index_in_block = threadIdx.x;
	int total_index_in_group = threadIdx.x + blockDim.x * blockIdx.x;

	int kernel_group_offset = blockIdx.y * K;
	int input_data_group_offset = blockIdx.y * N * K;
	int output_data_group_offset = blockIdx.y * N;

	if(thread_index_in_block < K) {
		x[thread_index_in_block] = x_gpu[kernel_group_offset + thread_index_in_block];
	}

	__syncthreads();

	float acc = 0.0;

	if(total_index_in_group < N) {
		for(int i = 0; i < K; i++) {
			acc += x[i] * A_gpu[input_data_group_offset + i*N + total_index_in_group];
		}
		C_gpu[output_data_group_offset + total_index_in_group] = acc;
	}

}


//zerotech Single float matrix-vector multiply
void ztSgmvStrideBatched(const float * __restrict__ x_gpu, int K,
		const float * __restrict__ A_gpu, int N, int groups, float * __restrict__ C_gpu) {

	if(K == 9) {
		ztSgmvStrideBatched_3x3<<< dim3((N + BLOCK -1)/BLOCK, groups, 1) , BLOCK>>>(x_gpu, K,
				A_gpu, N, groups, C_gpu);
	}
	else {
		ztSgmvStrideBatched_nxn<<< dim3((N + BLOCK -1)/BLOCK, groups, 1) , BLOCK>>>(x_gpu, K,
				A_gpu, N, groups, C_gpu);
	}

}
