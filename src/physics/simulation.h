#pragma once

#include "particle_system.h"
#include "cuda/morton.cuh"
#include "cuda/octree.cuh"
#include "cuda/density_field.cuh"
#include "cuda/diagnostics.cuh"

struct SimulationConfig {
    float G, softening, timestep, theta, mergeDistance;
    bool bruteForce, evolution, adaptiveTimestep, adaptiveTheta;
    bool volumetricEnabled;
    int densityFieldRes;
};

class Simulation {
public:
    Simulation();
    ~Simulation();
    void init(const SimulationConfig& cfg, uint32_t count);
    void cleanup();
    void setParticles(ParticleSystem* ps);
    void step();
    void pause() { m_paused = true; }
    void resume() { m_paused = false; }
    bool isPaused() const { return m_paused; }
    float getTime() const { return m_time; }
    uint64_t getStep() const { return m_step; }
    float treeBuildMs() const { return m_treeMs; }
    float forceCalcMs() const { return m_forceMs; }
    float integrationMs() const { return m_intMs; }
    float totalStepMs() const { return m_totalMs; }
    SimulationConfig& config() { return m_cfg; }
    const SimDiagnostics& diagnostics() const { return m_diag; }
    const DensityField& densityField() const { return m_densityField; }
    const OctreeData& octree() const { return m_octree; }

private:
    SimulationConfig m_cfg;
    ParticleSystem* m_ps;
    MortonData m_morton;
    OctreeData m_octree;
    DensityField m_densityField;
    SimDiagnostics m_diag;
    bool m_paused;
    float m_time;
    uint64_t m_step;
    cudaStream_t m_physicsStream;
    cudaStream_t m_auxStream;
    bool m_allocated;
    cudaEvent_t m_ev[5];
    float m_treeMs, m_forceMs, m_intMs, m_totalMs;
};