#pragma once

#include <cuda_runtime.h>
#include "kernels.cuh"
#include "octree.cuh"

void launchBruteForce(ParticleDeviceData& d, float G, float eps, uint32_t n, cudaStream_t s);
void launchBarnesHut(ParticleDeviceData& d, const OctreeData& tree, const uint32_t* si, float G, float eps, float theta, uint32_t n, cudaStream_t s);
void launchIntegrate(ParticleDeviceData& d, float dt, uint32_t n, cudaStream_t s);
void launchCollisions(ParticleDeviceData& d, float mergeDist, uint32_t n, cudaStream_t s);