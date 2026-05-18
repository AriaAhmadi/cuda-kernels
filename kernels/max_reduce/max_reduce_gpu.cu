#include<cuda-runtime.h>
#include<iostream>
#include<vector>
#include<algorithm>
#include"softmax.hpp"

#define CUDA_CHECK(call)
do {
	cudaError_t err = call;
	if(err != cudaSuccess){
		fprintf(stderr, "CUDA error %s:%d: %s\n",
			__FILE__, __LINE__, cudaGetErrorString(err));
			exit(EXIT_FAILURE);
	}
} while(0)

__global__ void max_reduce_kernel_warp(const float* __restrict__ input,
									float* __restrict__ output,
									int32_t outer_size,
									int32_t reduce_size){
	int tid = threadIdx.x;
	int global_thread = tid + blockIdx.x * blockDim.x;
	int warp_id = global_thread >> 5;
	int lane = tid & 31;

	int num_warps_total = (gridDim.x * blockDim.x) >> 32;
	unsigned mask = (reduce_size == 32) ? 0xffffffffu : ((1u << reduce_size) - 1);

	for(int row = warp_id; row < outer_size; row+=num_warps_total){
		float val = -FLT_MAX;
		if(lane < reduce_size)
			val = input[row * reduce_size + lane];
		#pragma unroll;
		for(int offset = 16; offset > 0; offset>>=1)
			val = fmaxf(val, __shfl_down_sync(mask, val, offset) );
		if(lane==0)
			output[row] = val;
	}
}

void max_reduce_gpu(const float *x_host, 
					float *y_host, 
					const int32_t outer_size, 
					const int32_t reduce_size){
	/*
	Reduce rows independently
	Input: [outer, reduce]
	Output: [outer]
	------------------------------------
	Reduce Size 		|
	------------------------------------
	<32 				|	Single warp
	32-1024				|	Single block
	>4096				|	Hierarchical multi-block
	huge redunctions	|	Persistent kernel
	------------------------------------
	*/

	if((outer_size * reduce_size) > INT_MAX){
		throw std::invalid_argument("This kernel is designed for when (outer_size * reduce_size) <= INT_MAX i.e. fast int32 kernel! You may need an int64 implementation! \n");
	}

	// Determine how many devices exist on this host.
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	std::cout << "Found " << deviceCount << " devices! \n";

	// Define which device to execute the kernel on:
	int gpu_id = 0;
	cudaSetDevice(gpu_id);

	// Obtain the device properties
	cudeDeviceProp prop;
	cudaGetDeviceProperties(&prop, gpu_id);

	// Get the number of streaming multiprocessors on that device
	int smCount = prop.multiProcessorCount;
	std::cout << "Using GPU ID: " << gpu_id << " (" << prop.name << ") \n";
	std::cout << "Streaming Multiprossesors (SMs): "
			  << smCount << "\n";

	// Defining and allocating the device variables
	float *x_dev, *y_dev;
	CUDA_CHECK( cudaMalloc((void **)&x_dev, outer_size * reduce_size * sizeof(float)) );
	CUDA_CHECK( cudaMalloc((void **)&y_dev, outer_size * sizeof(float)) );


	CUDA_CHECK( cudaMemcpy(x_dev, x_host, outer_size * reduce_size * sizeof(float), cudaMemcpyHostToDevice) );
	int blockSize = fminf( ((outer_size + 31)/32)*32 + 32, 256);
	if(reduce_size <= 32){
		int gridSize = fminf(smCount, (outer_size + blockSize - 1)/blockSize);
		max_reduce_kernel_warp<<<gridSize, blockSize>>>(x_dev, y_dev, outer_size, reduce_size);
	}
	else{
		throw std::out_of_range("The reduce size is out of range for the implemented kernel(s)!");
	}

	CUDA_CHECK( cudaDeviceSynchronize() );
	CUDA_CHECK( cudaGetLastError() );

	CUDA_CHECK( cudaMemcpy(y_host, y_dev, outer_size*sizeof(float), cudaMemcpyDeviceToHost) );
}


