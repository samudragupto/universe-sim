#include "particle_system.h"
#include "cuda/cuda_utils.h"
#include <cstring>

ParticleSystem::ParticleSystem() : m_count(0), m_allocated(false) {
    memset(&m_data, 0, sizeof(m_data));
}

ParticleSystem::~ParticleSystem() { deallocate(); }

void ParticleSystem::allocate(uint32_t c) {
    if (m_allocated) deallocate();
    m_count = c; 
    m_data.count = c; 
    size_t fb = c * sizeof(float);
    size_t ub = c * sizeof(uint8_t);
    CUDA_CHECK(cudaMalloc(&m_data.pos_x, fb));
    CUDA_CHECK(cudaMalloc(&m_data.pos_y, fb));
    CUDA_CHECK(cudaMalloc(&m_data.pos_z, fb));
    CUDA_CHECK(cudaMalloc(&m_data.vel_x, fb));
    CUDA_CHECK(cudaMalloc(&m_data.vel_y, fb));
    CUDA_CHECK(cudaMalloc(&m_data.vel_z, fb));
    CUDA_CHECK(cudaMalloc(&m_data.acc_x, fb));
    CUDA_CHECK(cudaMalloc(&m_data.acc_y, fb));
    CUDA_CHECK(cudaMalloc(&m_data.acc_z, fb));
    CUDA_CHECK(cudaMalloc(&m_data.mass, fb));
    CUDA_CHECK(cudaMalloc(&m_data.radius, fb));
    CUDA_CHECK(cudaMalloc(&m_data.temperature, fb));
    CUDA_CHECK(cudaMalloc(&m_data.luminosity, fb));
    CUDA_CHECK(cudaMalloc(&m_data.age, fb));
    CUDA_CHECK(cudaMalloc(&m_data.type, ub));
    CUDA_CHECK(cudaMalloc(&m_data.alive, ub));
    m_allocated = true;
}

void ParticleSystem::deallocate() {
    if (!m_allocated) return;
    cudaFree(m_data.pos_x); cudaFree(m_data.pos_y); cudaFree(m_data.pos_z);
    cudaFree(m_data.vel_x); cudaFree(m_data.vel_y); cudaFree(m_data.vel_z);
    cudaFree(m_data.acc_x); cudaFree(m_data.acc_y); cudaFree(m_data.acc_z);
    cudaFree(m_data.mass); cudaFree(m_data.radius);
    cudaFree(m_data.temperature); cudaFree(m_data.luminosity); cudaFree(m_data.age);
    cudaFree(m_data.type); cudaFree(m_data.alive);
    memset(&m_data, 0, sizeof(m_data));
    m_count = 0; 
    m_allocated = false;
}

void ParticleSystem::initRandom(unsigned long long seed) {
    launchInitRandom(m_data, seed, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
}