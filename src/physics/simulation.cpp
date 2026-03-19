#include "simulation.h"
#include "cuda/cuda_utils.h"
#include "cuda/kernels.cuh"
#include "cuda/physics.cuh"
#include "cuda/stellar_evolution.cuh"
#include <algorithm>

Simulation::Simulation()
    : m_ps(nullptr),
      m_paused(true),
      m_time(0.0f),
      m_step(0),
      m_physicsStream(0),
      m_auxStream(0),
      m_allocated(false),
      m_treeMs(0.0f),
      m_forceMs(0.0f),
      m_intMs(0.0f),
      m_totalMs(0.0f) {
    m_cfg = {
        1.0f,
        0.1f,
        0.005f,
        1.15f,
        0.1f,
        false,
        false,
        true,
        true,
        true,
        2
    };
    memset(&m_diag, 0, sizeof(m_diag));
}

Simulation::~Simulation() {
    cleanup();
}

void Simulation::init(const SimulationConfig& cfg, uint32_t count) {
    m_cfg = cfg;

    CUDA_CHECK(cudaStreamCreate(&m_physicsStream));
    CUDA_CHECK(cudaStreamCreate(&m_auxStream));

    if (count > 1 && !m_cfg.bruteForce) {
        mortonAllocate(m_morton, count);
        octreeAllocate(m_octree, count);
    }

    m_allocated = true;

    for (int i = 0; i < 5; i++) {
        CUDA_CHECK(cudaEventCreate(&m_ev[i]));
    }
}

void Simulation::cleanup() {
    if (!m_allocated) return;

    if (!m_cfg.bruteForce) {
        mortonDeallocate(m_morton);
        octreeDeallocate(m_octree);
    }

    if (m_physicsStream) cudaStreamDestroy(m_physicsStream);
    if (m_auxStream) cudaStreamDestroy(m_auxStream);

    for (int i = 0; i < 5; i++) {
        cudaEventDestroy(m_ev[i]);
    }

    m_physicsStream = 0;
    m_auxStream = 0;
    m_allocated = false;
}

void Simulation::setParticles(ParticleSystem* ps) {
    m_ps = ps;
}

void Simulation::step() {
    if (m_paused || !m_ps || m_ps->count() < 2) return;

    uint32_t n = m_ps->count();
    auto& d = m_ps->deviceData();

    CUDA_CHECK(cudaEventRecord(m_ev[0], m_physicsStream));

    launchClearAccelerations(d, m_physicsStream);

    bool rebuildTree = !m_cfg.bruteForce;
    if (m_cfg.skipTreeRebuild && m_cfg.treeRebuildInterval > 1) {
        rebuildTree = rebuildTree && ((m_step % (uint64_t)m_cfg.treeRebuildInterval) == 0);
    }

    if (rebuildTree) {
        launchBoundingBox(
            d.pos_x, d.pos_y, d.pos_z,
            m_octree.bboxMin, m_octree.bboxMax,
            n, m_physicsStream);

        launchMortonCodes(
            d.pos_x, d.pos_y, d.pos_z,
            m_octree.bboxMin, m_octree.bboxMax,
            m_morton.codes, m_morton.indices,
            n, m_physicsStream);

        launchRadixSort(m_morton, m_physicsStream);

        launchBuildTree(
            m_morton.codes,
            m_octree,
            n,
            m_physicsStream);

        launchNodeProperties(
            d.pos_x, d.pos_y, d.pos_z,
            d.mass,
            m_morton.indices,
            m_octree,
            n,
            m_physicsStream);
    }

    CUDA_CHECK(cudaEventRecord(m_ev[1], m_physicsStream));

    if (m_cfg.bruteForce) {
        launchBruteForce(d, m_cfg.G, m_cfg.softening, n, m_physicsStream);
    } else {
        launchBarnesHut(d, m_octree, m_morton.indices, m_cfg.G, m_cfg.softening, m_cfg.theta, n, m_physicsStream);
    }

    CUDA_CHECK(cudaEventRecord(m_ev[2], m_physicsStream));

    launchIntegrate(d, m_cfg.timestep, n, m_physicsStream);

    CUDA_CHECK(cudaEventRecord(m_ev[3], m_physicsStream));

    launchCollisions(d, m_cfg.mergeDistance, n, m_physicsStream);

    if (m_cfg.evolution) {
        launchStellarEvolution(d, m_cfg.timestep, n, m_physicsStream);
    }

    if ((m_step % 20) == 0) {
        launchComputeDiagnostics(d, &m_diag, m_cfg.G, n, m_auxStream);
    }

    CUDA_CHECK(cudaEventRecord(m_ev[4], m_physicsStream));
    CUDA_CHECK(cudaStreamSynchronize(m_physicsStream));
    CUDA_CHECK(cudaStreamSynchronize(m_auxStream));

    CUDA_CHECK(cudaEventElapsedTime(&m_treeMs, m_ev[0], m_ev[1]));
    CUDA_CHECK(cudaEventElapsedTime(&m_forceMs, m_ev[1], m_ev[2]));
    CUDA_CHECK(cudaEventElapsedTime(&m_intMs, m_ev[2], m_ev[3]));
    CUDA_CHECK(cudaEventElapsedTime(&m_totalMs, m_ev[0], m_ev[4]));

    if (m_cfg.adaptiveTheta) {
        if (m_totalMs > 16.0f) m_cfg.theta += 0.02f;
        else if (m_totalMs < 10.0f) m_cfg.theta -= 0.01f;
        m_cfg.theta = std::clamp(m_cfg.theta, 0.8f, 1.5f);
    }

    if (m_cfg.adaptiveTimestep && m_diag.maxAcceleration > 0.0f) {
        float targetDt = 0.06f * sqrtf(m_cfg.softening / m_diag.maxAcceleration);
        m_cfg.timestep = m_cfg.timestep * 0.9f + targetDt * 0.1f;
        m_cfg.timestep = std::clamp(m_cfg.timestep, 0.0005f, 0.02f);
    }

    m_time += m_cfg.timestep;
    m_step++;
}