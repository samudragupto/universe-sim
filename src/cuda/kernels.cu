#include "kernels.cuh"
#include "cuda_utils.h"
#include <curand_kernel.h>
#include <math.h>

__device__ float clamp01(float x) {
    return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
}

__device__ void temperatureToRGB(float tempK, float& r, float& g, float& b) {
    float t = tempK / 12000.0f;

    if (t < 0.35f) {
        r = 1.0f;
        g = 0.35f + t * 0.9f;
        b = 0.18f + t * 0.15f;
    } else if (t < 0.7f) {
        float u = (t - 0.35f) / 0.35f;
        r = 1.0f - u * 0.15f;
        g = 0.65f + u * 0.25f;
        b = 0.35f + u * 0.50f;
    } else {
        float u = (t - 0.7f) / 0.8f;
        r = 0.85f - u * 0.25f;
        g = 0.90f - u * 0.10f;
        b = 1.0f;
    }

    r = clamp01(r);
    g = clamp01(g);
    b = clamp01(b);
}

__global__ void initRandomKernel(
    float* pos_x, float* pos_y, float* pos_z,
    float* vel_x, float* vel_y, float* vel_z,
    float* acc_x, float* acc_y, float* acc_z,
    float* m, float* radius, float* temp,
    float* lum, float* age,
    uint8_t* typ, uint8_t* alive,
    uint32_t count,
    unsigned long long seed
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    curandState state;
    curand_init(seed, i, 0, &state);

    float rad = curand_uniform(&state) * 50.0f;
    float theta = curand_uniform(&state) * 6.28318530718f;
    float u = curand_uniform(&state) * 2.0f - 1.0f;
    float phi = acosf(u);

    pos_x[i] = rad * sinf(phi) * cosf(theta);
    pos_y[i] = rad * sinf(phi) * sinf(theta);
    pos_z[i] = rad * cosf(phi);

    vel_x[i] = 0.0f;
    vel_y[i] = 0.0f;
    vel_z[i] = 0.0f;

    acc_x[i] = 0.0f;
    acc_y[i] = 0.0f;
    acc_z[i] = 0.0f;

    m[i] = 1.0f;
    radius[i] = 0.01f;
    temp[i] = 3000.0f + curand_uniform(&state) * 27000.0f;
    lum[i] = 0.4f + curand_uniform(&state) * 1.6f;
    age[i] = 0.0f;
    typ[i] = PARTICLE_STAR;
    alive[i] = 1;
}

__global__ void computeColorsKernel(
    const float* pos_x, const float* pos_y, const float* pos_z,
    const float* temperature, const float* luminosity,
    const uint8_t* type, const uint8_t* alive,
    RenderVertex* out,
    uint32_t count
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    RenderVertex v;
    v.px = pos_x[i];
    v.py = pos_y[i];
    v.pz = pos_z[i];

    if (!alive[i]) {
        v.cr = 0.0f;
        v.cg = 0.0f;
        v.cb = 0.0f;
        v.ca = 0.0f;
        v.size = 0.0f;
        out[i] = v;
        return;
    }

    uint8_t t = type[i];

    if (t == PARTICLE_STAR) {
        float r, g, b;
        temperatureToRGB(temperature[i], r, g, b);
        float l = fminf(luminosity[i], 6.0f);

        // subtle brightness boost
        float brightness = 0.85f + l * 0.12f;

        v.cr = clamp01(r * brightness);
        v.cg = clamp01(g * brightness);
        v.cb = clamp01(b * brightness);
        v.ca = 0.85f;
        v.size = 0.45f + l * 0.06f;
    }
    else if (t == PARTICLE_GAS) {
        v.cr = 1.0f;
        v.cg = 0.45f;
        v.cb = 0.18f;
        v.ca = 0.14f;
        v.size = 0.75f;
    }
    else if (t == PARTICLE_BH) {
        v.cr = 1.0f;
        v.cg = 0.72f;
        v.cb = 0.22f;
        v.ca = 1.0f;
        v.size = 1.5f;
    }
    else if (t == PARTICLE_DM) {
        v.cr = 0.45f;
        v.cg = 0.28f;
        v.cb = 0.85f;
        v.ca = 0.04f;
        v.size = 0.4f;
    }
    else if (t == PARTICLE_NS) {
        v.cr = 0.55f;
        v.cg = 0.85f;
        v.cb = 1.0f;
        v.ca = 0.95f;
        v.size = 0.7f;
    }
    else {
        v.cr = 1.0f;
        v.cg = 1.0f;
        v.cb = 1.0f;
        v.ca = 0.8f;
        v.size = 0.5f;
    }

    out[i] = v;
}

__global__ void clearAccelerationsKernel(
    float* acc_x, float* acc_y, float* acc_z,
    uint32_t count
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    acc_x[i] = 0.0f;
    acc_y[i] = 0.0f;
    acc_z[i] = 0.0f;
}

void launchInitRandom(ParticleDeviceData& d, unsigned long long seed, cudaStream_t s) {
    initRandomKernel<<<divUp((int)d.count, 256), 256, 0, s>>>(
        d.pos_x, d.pos_y, d.pos_z,
        d.vel_x, d.vel_y, d.vel_z,
        d.acc_x, d.acc_y, d.acc_z,
        d.mass, d.radius, d.temperature,
        d.luminosity, d.age,
        d.type, d.alive,
        d.count,
        seed
    );
    CUDA_CHECK(cudaGetLastError());
}

void launchComputeColors(ParticleDeviceData& d, RenderVertex* buf, uint32_t n, cudaStream_t s) {
    computeColorsKernel<<<divUp((int)n, 256), 256, 0, s>>>(
        d.pos_x, d.pos_y, d.pos_z,
        d.temperature, d.luminosity,
        d.type, d.alive,
        buf,
        n
    );
    CUDA_CHECK(cudaGetLastError());
}

void launchClearAccelerations(ParticleDeviceData& d, cudaStream_t s) {
    clearAccelerationsKernel<<<divUp((int)d.count, 256), 256, 0, s>>>(
        d.acc_x, d.acc_y, d.acc_z,
        d.count
    );
    CUDA_CHECK(cudaGetLastError());
}