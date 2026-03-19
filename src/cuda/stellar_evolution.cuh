#pragma once
#include <cuda_runtime.h>
#include "kernels.cuh"
void launchStellarEvolution(ParticleDeviceData& d, float dt, uint32_t n, cudaStream_t s);
