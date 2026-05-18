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

#define outer_size 150
#define reduce_size 22

int main(void){
	std::vector<float> x_host(outer_size * reduce_size), y_host(outer_size);

	// Initialise the input
	for(int i = 0; i < outer_size * reduce_size; i++){
		x_host[i] = i;
	}

	// Defining the profiling events
	cudaEvent_t start, stop;
	CUDA_CHECK( cudaEventCreate(&start) );
	CUDA_CHECK( cudaEventCreate(&stop) );

	CUDA_CHECK( cudaEventRecord(start, 0) );
	// ===============================
	max_reduce_gpu(x_host.data(), y_host.data(), outer_size, reduce_size);
	// ===============================

	CUDA_CHECK( cudaEventRecord(stop, 0) );
	CUDA_CHECK( cudaEventSynchronize( stop ));

	// Measure and report the runtime
	float ealpsed_time;
	CUDA_CHECK( cudaEventElapsedTime(&ealpsed_time, start, stop) );
	printf("The runtime is: %3.1f ms \n", ealpsed_time);
	CUDA_CHECK( cudaEventDestroy(start));
	CUDA_CHECK( cudaEventDestroy(stop));

	// Print a few output elements (sanity check)
	std::cout << "Sample output";
	for(int i = 0; i<15; i++)
		std::cout << y_host[i] << " ";
	std::cout << std::endl;

	return 0;
}