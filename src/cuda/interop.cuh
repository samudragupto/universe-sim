#pragma once

#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "kernels.cuh"

class CUDAGLInterop {
public:
    CUDAGLInterop();
    ~CUDAGLInterop();
    void registerBuffer(unsigned int glBuffer, uint32_t particleCount);
    void unregisterBuffer();
    RenderVertex* mapBuffer(cudaStream_t stream = 0);
    void unmapBuffer(cudaStream_t stream = 0);
    bool isRegistered() const { return m_registered; }
private:
    cudaGraphicsResource_t m_resource;
    bool m_registered;
    bool m_mapped;
    uint32_t m_count;
};