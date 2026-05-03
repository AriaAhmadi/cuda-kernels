#include"correlation_1d.hpp"
#include<iostream>
#include<vector>
#include<cstdlib>
#include<cuda_runtime.h>


#define CUDA_CHECK(call)
do {
	cudaError_t err = call;
	if(err != cudaSuccess){
		fprintf(stderr, "CUDA error %s:%d: %s\n",
			__FILE__, __LINE__, cudaGetErrorString(err));
			exit(EXIT_FAILURE);
	}
} while(0)

int main(void){
	cudaEvent+t start, stop;
	CUDA_CHECK( cudaEventCreate(&start) );
	CUDA_CHECK( cudaEventCreate(&stop) );

	int N = 2048;
	int F = 5;

	float *x_host, *y_host, *f_host;
	x_host = (float *)malloc(N*sizeof(float));
	f_host = (float *)malloc(F*sizeof(float));
	for(int i = 0; i< N; i++){
		x_host[i] = rand();
		if(i<F){
			f_host[i] = rand();
		}
	}
	y_host = (float *)malloc(N*sizeof(float));

	CUDA_CHECK( cudaEventRecord(start, 0) );
	correlation_1d_gpu(x_host, f_host, y_host, N, F);

	CUDA_CHECK( cudaEventRecord(stop, 0) );
	CUDA_CHECK( cudaEventSynchronize(stop) );
	float elapsed_time;
	CUDA_CHECK( cudaEventElapsedTime(&elapsed_time, start, stop) );
	printf("The execution time has taken %3.2f ms \n", elapsed_time);

	std::cout << "Sample output: ";
	for(int i = 0; i<5; i++){
		std::cout << y_host[i] << " ";
	}
	std::cout << std::endl;

	CUDA_CHECK( cudaEventDestroy(start) );
	CUDA_CHECK( cudaEventDestroy(stop) );
	return 0;
}