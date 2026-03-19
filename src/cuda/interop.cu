#include "interop.cuh"
#include "cuda_utils.h"
CUDAGLInterop::CUDAGLInterop() : m_resource(nullptr), m_registered(false), m_mapped(false), m_count(0) {}
CUDAGLInterop::~CUDAGLInterop() { if(m_mapped) unmapBuffer(); if(m_registered) unregisterBuffer(); }
void CUDAGLInterop::registerBuffer(unsigned int buf, uint32_t c) { if(m_registered) unregisterBuffer(); CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_resource, buf, cudaGraphicsMapFlagsWriteDiscard)); m_registered=true; m_count=c; }
void CUDAGLInterop::unregisterBuffer() { if(!m_registered) return; if(m_mapped) unmapBuffer(); CUDA_CHECK_NOEXIT(cudaGraphicsUnregisterResource(m_resource)); m_registered=false; }
RenderVertex* CUDAGLInterop::mapBuffer(cudaStream_t s) { if(!m_registered||m_mapped) return nullptr; CUDA_CHECK(cudaGraphicsMapResources(1, &m_resource, s)); RenderVertex* ptr=nullptr; size_t sz=0; CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&ptr, &sz, m_resource)); m_mapped=true; return ptr; }
void CUDAGLInterop::unmapBuffer(cudaStream_t s) { if(!m_mapped) return; CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_resource, s)); m_mapped=false; }