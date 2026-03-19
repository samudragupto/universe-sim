#pragma once

#include <cuda_runtime.h>
#include <cstdint>

struct GPUProfiler {
    cudaEvent_t events[16];
    const char* names[8];
    float times[8];
    int count;

    void init();
    void destroy();
    void begin(int slot, const char* name, cudaStream_t stream);
    void end(int slot, cudaStream_t stream);
    void resolve();
};