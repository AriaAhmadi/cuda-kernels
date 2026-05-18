#pragma once

void max_reduce_cpu(const float *x, float *y, const int32_t outer_size, const int32_t reduce_size);
void max_reduce_gpu(const float *x, float *y, const int32_t outer_size, const int32_t reduce_size);