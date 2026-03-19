#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { cudaError_t e = call; if(e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)
#define CUDA_CHECK_NOEXIT(call) do { cudaError_t e = call; if(e != cudaSuccess) { \
    fprintf(stderr, "CUDA warn %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); } } while(0)

inline int divUp(int n, int d) { return (n + d - 1) / d; }