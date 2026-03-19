#pragma once

#include <cuda_runtime.h>
#include "kernels.cuh"

struct LODResult {
    uint32_t* visibleIndices;
    uint32_t* visibleCount;
    uint32_t maxVisible;
    bool allocated;
};

void lodAllocate(LODResult& lod, uint32_t maxParticles);
void lodDeallocate(LODResult& lod);
void launchLODCull(
    const ParticleDeviceData& particles,
    LODResult& lod,
    float camX, float camY, float camZ,
    const float* frustumPlanes,
    float lodDistances[4],
    uint32_t count, cudaStream_t stream);