#pragma once
#include <cuda_runtime.h>
#include <cstdint>
enum ParticleType : uint8_t { PARTICLE_STAR=0, PARTICLE_DM=1, PARTICLE_BH=2, PARTICLE_GAS=3, PARTICLE_NS=4 };
struct ParticleDeviceData {
    float *pos_x, *pos_y, *pos_z, *vel_x, *vel_y, *vel_z, *acc_x, *acc_y, *acc_z;
    float *mass, *radius, *temperature, *luminosity, *age;
    uint8_t *type, *alive;
    uint32_t count;
};
struct RenderVertex { float px, py, pz, cr, cg, cb, ca, size; };
void launchInitRandom(ParticleDeviceData& d, unsigned long long seed, cudaStream_t s);
void launchComputeColors(ParticleDeviceData& d, RenderVertex* buf, uint32_t count, cudaStream_t s);
void launchClearAccelerations(ParticleDeviceData& d, cudaStream_t s);