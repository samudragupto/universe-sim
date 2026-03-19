#pragma once
#include <cuda_runtime.h>
#include <cstdint>
struct MortonData { uint64_t* codes; uint32_t* indices; uint64_t* codesAlt; uint32_t* indicesAlt; void* temp; size_t tempBytes; uint32_t count; };
void mortonAllocate(MortonData& d, uint32_t count); void mortonDeallocate(MortonData& d);
void launchMortonCodes(const float* px, const float* py, const float* pz, const float* bmin, const float* bmax, uint64_t* codes, uint32_t* indices, uint32_t count, cudaStream_t stream);
void launchRadixSort(MortonData& d, cudaStream_t stream);
