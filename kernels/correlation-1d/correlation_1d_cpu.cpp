#include"correlation_1d.hpp"

void correlation_1d_cpu(const float* x,
						const float* f,
						float* y,
						int N,
						int F){
	int R = F/2;
	for(int i=0; i<N; i++){
		float sum = 0.0f;
		for(int k=-R; k<=R; k++){
			int idx = i + k;
			if(idx>=0 && idx<N){
				sum += x[idx] * f[k+R];
			}
		}
		y[i] = sum;
	}
}