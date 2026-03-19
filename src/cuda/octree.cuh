#pragma once

#include <cuda_runtime.h>
#include <cstdint>

struct OctreeData {
    float *com_x, *com_y, *com_z;
    float *total_mass;
    float *bmin_x, *bmin_y, *bmin_z;
    float *bmax_x, *bmax_y, *bmax_z;
    int *left, *right;
    int *particle_start, *particle_count;
    int *parentIdx;
    int *leftRange, *rightRange;
    int *atomic_visited;
    uint8_t *is_leaf;

    // Bounding box for the whole tree
    float *bboxMin;
    float *bboxMax;

    uint32_t numInternal;
    uint32_t numLeaves;
    uint32_t numNodes;
    uint32_t particleCount;
};

void octreeAllocate(OctreeData& d, uint32_t count);
void octreeDeallocate(OctreeData& d);
void launchBoundingBox(const float* px, const float* py, const float* pz,
                       float* bmin, float* bmax, uint32_t count, cudaStream_t s);
void launchBuildTree(const uint64_t* codes, OctreeData& tree, uint32_t n, cudaStream_t s);
void launchNodeProperties(const float* px, const float* py, const float* pz, const float* mass,
                          const uint32_t* sortedIdx, OctreeData& tree, uint32_t n, cudaStream_t s);