#pragma once

#include <cuda_runtime.h>
#include "kernels.cuh"

struct TrailVertex {
    float px, py, pz;
    float cr, cg, cb, ca;
};

struct TrailData {
    TrailVertex* vertices;
    uint32_t maxParticles;
    uint32_t historyLength;
    uint32_t currentFrame;
    uint32_t totalVertices;
    bool allocated;
};

void trailAllocate(TrailData& d, uint32_t maxP, uint32_t histLen);
void trailDeallocate(TrailData& d);
void launchUpdateTrails(const ParticleDeviceData& p, TrailData& t, uint32_t n, cudaStream_t s);