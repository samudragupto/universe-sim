#include "lod.cuh"
#include "cuda_utils.h"

void lodAllocate(LODResult& lod, uint32_t maxP) {
    lod.maxVisible = maxP;
    lod.allocated = true;
    CUDA_CHECK(cudaMalloc(&lod.visibleIndices, maxP * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&lod.visibleCount, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(lod.visibleCount, 0, sizeof(uint32_t)));
}

void lodDeallocate(LODResult& lod) {
    if (lod.allocated) {
        cudaFree(lod.visibleIndices);
        cudaFree(lod.visibleCount);
    }
    lod.allocated = false;
}

__global__ void lodCullKernel(
    const float* __restrict__ px, const float* __restrict__ py, const float* __restrict__ pz,
    const float* __restrict__ mass, const uint8_t* __restrict__ alive,
    uint32_t* __restrict__ visibleIdx, uint32_t* __restrict__ visibleCount,
    float camX, float camY, float camZ,
    float lod0, float lod1, float lod2, float lod3,
    uint32_t maxVisible, uint32_t count
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count || !alive[i]) return;

    float dx = px[i] - camX;
    float dy = py[i] - camY;
    float dz = pz[i] - camZ;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);

    bool visible = true;

    if (dist > lod3) {
        visible = (i % 16 == 0);
    } else if (dist > lod2) {
        visible = (i % 8 == 0);
    } else if (dist > lod1) {
        visible = (i % 4 == 0);
    } else if (dist > lod0) {
        visible = (i % 2 == 0);
    }

    if (mass[i] > 10.0f) visible = true;

    if (visible) {
        uint32_t slot = atomicAdd(visibleCount, 1);
        if (slot < maxVisible) {
            visibleIdx[slot] = i;
        }
    }
}

void launchLODCull(
    const ParticleDeviceData& p, LODResult& lod,
    float camX, float camY, float camZ,
    const float* frustumPlanes,
    float lodDist[4],
    uint32_t count, cudaStream_t stream
) {
    CUDA_CHECK(cudaMemsetAsync(lod.visibleCount, 0, sizeof(uint32_t), stream));
    lodCullKernel<<<divUp(count, 256), 256, 0, stream>>>(
        p.pos_x, p.pos_y, p.pos_z, p.mass, p.alive,
        lod.visibleIndices, lod.visibleCount,
        camX, camY, camZ,
        lodDist[0], lodDist[1], lodDist[2], lodDist[3],
        lod.maxVisible, count);
    CUDA_CHECK(cudaGetLastError());
}