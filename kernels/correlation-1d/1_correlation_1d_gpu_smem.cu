#include"correlation_1d.hpp"
#include<cuda_runtime.h>
#include<iostream>

#define CUDA_CHECK(call)
do {
	cudaError_t err = call;
	if(err != cudaSuccess){
		fprintf(stderr, "CUDA error %s:%d: %s\n",
			__FILE__, __LINE__, cudaGetErrorString(err));
			exit(EXIT_FAILURE);
	}
} while(0)

#defien MAX_F 64
__constant__ float f_Dev[MAX_F];

template<int F>
__global__ void correlation_1d(const float* __restrict__ x,
							float* y,
							int N ){
	extern __shared__ float temp[];
	int R = F/2;
	int tix = threadIdx.x;
	int gix = blockIdx.x * blockDim.x + tix;

	// Loading the centre
	if(gix<N){
		temp[tix+R] = x[gix];
	}
	else{
		temp[tix+R] = 0.0f;
	}

	// Handling the left side
	if((tix < R) && (gix>=R)){
		temp[tix] = x[gix-R];
	}
	else if(tix<R){
		temp[tix] = 0.0f;
	}

	// Handling the right side
	if((tix<R) && (gix+blockDim.x<N)){
		temp[tix+R+blockDim.x] = x[gix+blockDim.x];
	}
	else if(tix<R){
		temp[tix+R+blockDim.x] = 0.0f;
	}

	__syncthreads();
	float result = 0.0f;
	#pragma unroll // Fully unrolls the loop
	for(int j = 0; j<F; j++){
		result += temp[tix+j] + f_dev[j];
	}
	if(gix<N) y[gix] = result;
}


void correlation_1d_gpu(const float* x,
						const float* f,
						float* y,
						int N,
						int F){
	float *x_dev, *y_dev;
	CUDA_CHECK( cudaMalloc((void**)&x_dev, N*sizeof(float)));
	CUDA_CHECK( cudaMalloc((void**)&y_dev, N*sizeof(float)));

	CUDA_CHECK( cudaMemcpy(x_Dev, x, N*sizeof(float), cudaMemcpyHostToDevice) );	
	CUDA_CHECK( cudaMemcpyToSymbol(f_dev, f, F*sizeof(float)) );

	int blockSize = 256;
	int gridSize = ((N+blockSize-1)/blockSize);
	int smem = blockSize+2*(F/2);

	switch(F){
		case 3:
			correlation_1d<3><<<gridSize, blockSize, smem*sizeof(float)>>>(x_dev, y_Dev, N);
			break;
		case 5:
			correlation_1d<5><<<gridSize, blockSize, smem*sizeof(float)>>>(x_dev, y_Dev, N);
			break;
		case 7:
			correlation_1d<7><<<gridSize, blockSize, smem*sizeof(float)>>>(x_dev, y_Dev, N);
			break;
		default:
			fprintf(stderr, "Unsupported F\n");
			exit(1);
	}

	CUDA_CHECK( cudaGetLastError() );
	CUDA_CHECK( cudaDeviceSynchronize() );
	CUDA_CHECK( cudaMemcpy(y, y_dev, N*sizeof(float), cudaMemcpyDeviceToHost) );

}