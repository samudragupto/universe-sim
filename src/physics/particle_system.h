#pragma once

#include <cstdint>
#include "cuda/kernels.cuh"

class ParticleSystem {
public:
    ParticleSystem();
    ~ParticleSystem();
    void allocate(uint32_t count);
    void deallocate();
    void initRandom(unsigned long long seed);
    ParticleDeviceData& deviceData() { return m_data; }
    uint32_t count() const { return m_count; }
    bool isAllocated() const { return m_allocated; }
private:
    ParticleDeviceData m_data;
    uint32_t m_count;
    bool m_allocated;
};