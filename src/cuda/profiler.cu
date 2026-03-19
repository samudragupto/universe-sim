#include "profiler.cuh"
#include "cuda_utils.h"

void GPUProfiler::init() {
    count = 0;
    for (int i = 0; i < 16; i++) {
        CUDA_CHECK(cudaEventCreate(&events[i]));
    }
    for (int i = 0; i < 8; i++) {
        names[i] = nullptr;
        times[i] = 0.0f;
    }
}

void GPUProfiler::destroy() {
    for (int i = 0; i < 16; i++) {
        cudaEventDestroy(events[i]);
    }
}

void GPUProfiler::begin(int slot, const char* name, cudaStream_t stream) {
    if (slot >= 8) return;
    names[slot] = name;
    CUDA_CHECK(cudaEventRecord(events[slot * 2], stream));
    if (slot >= count) count = slot + 1;
}

void GPUProfiler::end(int slot, cudaStream_t stream) {
    if (slot >= 8) return;
    CUDA_CHECK(cudaEventRecord(events[slot * 2 + 1], stream));
}

void GPUProfiler::resolve() {
    for (int i = 0; i < count; i++) {
        CUDA_CHECK(cudaEventElapsedTime(&times[i], events[i * 2], events[i * 2 + 1]));
    }
}