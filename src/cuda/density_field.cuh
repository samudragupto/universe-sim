#pragma once

#include <cuda_runtime.h>
#include "kernels.cuh"

struct DensityField {
    float* data;
    int resX, resY, resZ;
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
    bool allocated;
};

void densityFieldAllocate(DensityField& df, int res);
void densityFieldDeallocate(DensityField& df);
void launchScatterDensity(const ParticleDeviceData& particles, DensityField& df,
                          const float* bboxMin, const float* bboxMax,
                          uint32_t count, cudaStream_t stream);