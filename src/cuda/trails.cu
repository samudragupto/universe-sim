#include "trails.cuh"
#include "cuda_utils.h"

void trailAllocate(TrailData& d, uint32_t mp, uint32_t hl) {
    d.maxParticles = mp;
    d.historyLength = hl;
    d.currentFrame = 0;
    d.totalVertices = mp * hl;
    d.allocated = true;
    CUDA_CHECK(cudaMalloc(&d.vertices, d.totalVertices * sizeof(TrailVertex)));
    CUDA_CHECK(cudaMemset(d.vertices, 0, d.totalVertices * sizeof(TrailVertex)));
}

void trailDeallocate(TrailData& d) {
    if(d.allocated && d.vertices) cudaFree(d.vertices);
    d.vertices = nullptr;
    d.allocated = false;
}

__global__ void uTK(const float* px, const float* py, const float* pz, const uint8_t* typ, const uint8_t* al, TrailVertex* v, uint32_t mp, uint32_t hl, uint32_t f, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; 
    if(i >= n || i >= mp) return;
    
    uint32_t s = (f % hl) * mp + i;
    if(!al[i]) { v[s].ca = 0.0f; return; }
    
    v[s].px = px[i]; v[s].py = py[i]; v[s].pz = pz[i];
    float a = 0.2f;
    if(typ[i] == 0) { v[s].cr = 1.0f; v[s].cg = 0.8f; v[s].cb = 0.5f; }
    else if(typ[i] == 2) { v[s].cr = 1.0f; v[s].cg = 0.3f; v[s].cb = 0.0f; a = 0.5f; }
    else { v[s].cr = 0.5f; v[s].cg = 0.5f; v[s].cb = 0.5f; a = 0.1f; }
    v[s].ca = a;
}

__global__ void fTK(TrailVertex* v, uint32_t tot, uint32_t mp, uint32_t cs, float fd) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= tot) return;
    if(i / mp != cs) {
        v[i].ca *= fd;
        if(v[i].ca < 0.001f) v[i].ca = 0.0f;
    }
}

void launchUpdateTrails(const ParticleDeviceData& p, TrailData& t, uint32_t n, cudaStream_t s) {
    uint32_t c = min(n, t.maxParticles);
    uTK<<<divUp(c, 256), 256, 0, s>>>(p.pos_x, p.pos_y, p.pos_z, p.type, p.alive, t.vertices, t.maxParticles, t.historyLength, t.currentFrame, c);
    fTK<<<divUp(t.totalVertices, 256), 256, 0, s>>>(t.vertices, t.totalVertices, t.maxParticles, t.currentFrame % t.historyLength, 0.95f);
    CUDA_CHECK(cudaGetLastError());
    t.currentFrame++;
}