#include"correlation_1d.hpp"
#include<iostream>
#include<vector>
#include<cmath>
#include<casssert>

bool almost_equal(float a, float b, float eps = 1e-5f){
	return std::fabs(a-b) < eps;
}

int main(void){
	int N = 2048;
	int F = 5;

	std::vector<float> x(N), f(F), y_cpu(N), y_gpu(N);

	for(int i = 0; i < N ; i++){
		x[i] = static_cast<float>(rand()) / RAND_MAX;
	}
	for(int i = 0; i < F ; i++){
		f[i] = static_cast<float>(rand()) / RAND_MAX;
	}

	correlation_1d_cpu(x.data(), f.data(), y_cpu.data(), N, F);
	correlation_1d_gpu(x.data(), f.data(), y_gpu.data(), N, F);

	for(int i = 0; i<N; i++){
		if(!almost_equal(y_cpu[i], y_gpu[i])){
			std::cerr << "Mismatch at " << i
					  << "CPU = " << y_cpu[i]
					  << "GPU = " << y_gpu[i] << "\n";
			return 1;
		}
	}

	std::cout << "Test passed! \n";
	return 0;
}