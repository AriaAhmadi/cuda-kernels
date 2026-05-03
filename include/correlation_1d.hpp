 #pragma once

void correlation_1d_cpu(const float* x,
						const float* f,
						float* y,
						int N,
						int F);

void correlation_1d_gpu(const float* x,
						const float* f,
						float* y,
						int N,
						int F);