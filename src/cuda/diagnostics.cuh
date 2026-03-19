#pragma once

#include <cuda_runtime.h>
#include "kernels.cuh"

struct SimDiagnostics {
    float totalKineticEnergy;
    float totalPotentialEnergy;
    float totalEnergy;
    float maxVelocity;
    float maxAcceleration;
    float comX, comY, comZ;
    uint32_t aliveCount;
    uint32_t starCount;
    uint32_t gasCount;
    uint32_t bhCount;
    uint32_t dmCount;
    uint32_t nsCount;
};

void launchComputeDiagnostics(const ParticleDeviceData& d, SimDiagnostics* hr, float G, uint32_t n, cudaStream_t s);