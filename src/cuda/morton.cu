#include "morton.cuh"
#include "cuda_utils.h"
#include <cub/cub.cuh>
__device__ uint32_t eB(uint32_t v) { v=(v*0x00010001u)&0xFF0000FFu; v=(v*0x00000101u)&0x0F00F00Fu; v=(v*0x00000011u)&0xC30C30C3u; return (v*0x00000005u)&0x49249249u; }
__device__ uint64_t m3D(float x, float y, float z) { uint32_t ix=x*1023.0f, iy=y*1023.0f, iz=z*1023.0f; return (uint64_t(eB(ix))<<2)|(uint64_t(eB(iy))<<1)|eB(iz); }
__global__ void mK(const float* px, const float* py, const float* pz, const float* bmin, const float* bmax, uint64_t* c, uint32_t* idx, uint32_t n) {
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float sx=bmax[0]-bmin[0], sy=bmax[1]-bmin[1], sz=bmax[2]-bmin[2];
    c[i]=m3D((px[i]-bmin[0])*(sx>1e-10f?1.f/sx:0), (py[i]-bmin[1])*(sy>1e-10f?1.f/sy:0), (pz[i]-bmin[2])*(sz>1e-10f?1.f/sz:0)); idx[i]=i;
}
void mortonAllocate(MortonData& d, uint32_t n) {
    d.count=n; CUDA_CHECK(cudaMalloc((void**)&d.codes, n*8)); CUDA_CHECK(cudaMalloc((void**)&d.indices, n*4));
    CUDA_CHECK(cudaMalloc((void**)&d.codesAlt, n*8)); CUDA_CHECK(cudaMalloc((void**)&d.indicesAlt, n*4));
    uint64_t *k0=d.codes, *k1=d.codesAlt; uint32_t *v0=d.indices, *v1=d.indicesAlt;
    cub::DoubleBuffer<uint64_t> kb(k0,k1); cub::DoubleBuffer<uint32_t> vb(v0,v1);
    d.temp=nullptr; d.tempBytes=0; cub::DeviceRadixSort::SortPairs(d.temp, d.tempBytes, kb, vb, n); CUDA_CHECK(cudaMalloc(&d.temp, d.tempBytes));
}
void mortonDeallocate(MortonData& d) { cudaFree(d.codes); cudaFree(d.indices); cudaFree(d.codesAlt); cudaFree(d.indicesAlt); cudaFree(d.temp); d.codes=nullptr; d.indices=nullptr; d.codesAlt=nullptr; d.indicesAlt=nullptr; d.temp=nullptr; }
void launchMortonCodes(const float* px, const float* py, const float* pz, const float* bmin, const float* bmax, uint64_t* c, uint32_t* i, uint32_t n, cudaStream_t s) { mK<<<divUp(n,256),256,0,s>>>(px,py,pz,bmin,bmax,c,i,n); CUDA_CHECK(cudaGetLastError()); }
void launchRadixSort(MortonData& d, cudaStream_t s) {
    uint64_t *k0=d.codes, *k1=d.codesAlt; uint32_t *v0=d.indices, *v1=d.indicesAlt;
    cub::DoubleBuffer<uint64_t> kb(k0,k1); cub::DoubleBuffer<uint32_t> vb(v0,v1); size_t tb=d.tempBytes;
    cub::DeviceRadixSort::SortPairs(d.temp, tb, kb, vb, d.count, 0, 30, s); CUDA_CHECK(cudaGetLastError());
    if(kb.Current()!=d.codes) CUDA_CHECK(cudaMemcpyAsync(d.codes, kb.Current(), d.count*8, cudaMemcpyDeviceToDevice, s));
    if(vb.Current()!=d.indices) CUDA_CHECK(cudaMemcpyAsync(d.indices, vb.Current(), d.count*4, cudaMemcpyDeviceToDevice, s));
}