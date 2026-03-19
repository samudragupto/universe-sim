#include "octree.cuh"
#include "cuda_utils.h"
#include <cfloat>

void octreeAllocate(OctreeData& d, uint32_t count) {
    d.particleCount = count;
    d.numLeaves = count;
    d.numInternal = count > 1 ? count - 1 : 0;
    d.numNodes = d.numInternal + d.numLeaves;

    size_t fSize = d.numNodes * sizeof(float);
    size_t iSize = d.numNodes * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d.com_x, fSize));
    CUDA_CHECK(cudaMalloc(&d.com_y, fSize));
    CUDA_CHECK(cudaMalloc(&d.com_z, fSize));
    CUDA_CHECK(cudaMalloc(&d.total_mass, fSize));
    CUDA_CHECK(cudaMalloc(&d.bmin_x, fSize));
    CUDA_CHECK(cudaMalloc(&d.bmin_y, fSize));
    CUDA_CHECK(cudaMalloc(&d.bmin_z, fSize));
    CUDA_CHECK(cudaMalloc(&d.bmax_x, fSize));
    CUDA_CHECK(cudaMalloc(&d.bmax_y, fSize));
    CUDA_CHECK(cudaMalloc(&d.bmax_z, fSize));

    CUDA_CHECK(cudaMalloc(&d.left, iSize));
    CUDA_CHECK(cudaMalloc(&d.right, iSize));
    CUDA_CHECK(cudaMalloc(&d.particle_start, iSize));
    CUDA_CHECK(cudaMalloc(&d.particle_count, iSize));
    CUDA_CHECK(cudaMalloc(&d.parentIdx, iSize));
    CUDA_CHECK(cudaMalloc(&d.leftRange, iSize));
    CUDA_CHECK(cudaMalloc(&d.rightRange, iSize));
    CUDA_CHECK(cudaMalloc(&d.atomic_visited, iSize));

    CUDA_CHECK(cudaMalloc(&d.is_leaf, d.numNodes * sizeof(uint8_t)));

    CUDA_CHECK(cudaMalloc(&d.bboxMin, 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.bboxMax, 3 * sizeof(float)));
}

void octreeDeallocate(OctreeData& d) {
    if (d.com_x) cudaFree(d.com_x);
    if (d.com_y) cudaFree(d.com_y);
    if (d.com_z) cudaFree(d.com_z);
    if (d.total_mass) cudaFree(d.total_mass);
    if (d.bmin_x) cudaFree(d.bmin_x);
    if (d.bmin_y) cudaFree(d.bmin_y);
    if (d.bmin_z) cudaFree(d.bmin_z);
    if (d.bmax_x) cudaFree(d.bmax_x);
    if (d.bmax_y) cudaFree(d.bmax_y);
    if (d.bmax_z) cudaFree(d.bmax_z);
    if (d.left) cudaFree(d.left);
    if (d.right) cudaFree(d.right);
    if (d.particle_start) cudaFree(d.particle_start);
    if (d.particle_count) cudaFree(d.particle_count);
    if (d.parentIdx) cudaFree(d.parentIdx);
    if (d.leftRange) cudaFree(d.leftRange);
    if (d.rightRange) cudaFree(d.rightRange);
    if (d.atomic_visited) cudaFree(d.atomic_visited);
    if (d.is_leaf) cudaFree(d.is_leaf);
    if (d.bboxMin) cudaFree(d.bboxMin);
    if (d.bboxMax) cudaFree(d.bboxMax);
}

__global__ void initBBoxKernel(float* bmin, float* bmax) {
    bmin[0] = bmin[1] = bmin[2] = 1e30f;
    bmax[0] = bmax[1] = bmax[2] = -1e30f;
}

__global__ void bboxKernel(const float* __restrict__ px, const float* __restrict__ py,
                           const float* __restrict__ pz, float* bmin, float* bmax, uint32_t n) {
    __shared__ float smin[3][256], smax[3][256];
    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * blockDim.x + tid;
    uint32_t stride = blockDim.x * gridDim.x;

    float lmin[3] = {1e30f, 1e30f, 1e30f};
    float lmax[3] = {-1e30f, -1e30f, -1e30f};

    for (uint32_t j = i; j < n; j += stride) {
        float x = px[j], y = py[j], z = pz[j];
        lmin[0] = fminf(lmin[0], x); lmin[1] = fminf(lmin[1], y); lmin[2] = fminf(lmin[2], z);
        lmax[0] = fmaxf(lmax[0], x); lmax[1] = fmaxf(lmax[1], y); lmax[2] = fmaxf(lmax[2], z);
    }

    for (int d = 0; d < 3; d++) { smin[d][tid] = lmin[d]; smax[d][tid] = lmax[d]; }
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            for (int d = 0; d < 3; d++) {
                smin[d][tid] = fminf(smin[d][tid], smin[d][tid + s]);
                smax[d][tid] = fmaxf(smax[d][tid], smax[d][tid + s]);
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int d = 0; d < 3; d++) {
            atomicMin((int*)&bmin[d], __float_as_int(smin[d][0]));
            atomicMax((int*)&bmax[d], __float_as_int(smax[d][0]));
        }
    }
}

__global__ void padBBoxKernel(float* bmin, float* bmax) {
    for (int d = 0; d < 3; d++) {
        float center = (bmin[d] + bmax[d]) * 0.5f;
        float half = (bmax[d] - bmin[d]) * 0.5f + 1.0f;
        bmin[d] = center - half;
        bmax[d] = center + half;
    }
}

void launchBoundingBox(const float* px, const float* py, const float* pz,
                       float* bmin, float* bmax, uint32_t n, cudaStream_t s) {
    initBBoxKernel<<<1, 1, 0, s>>>(bmin, bmax);
    int blocks = min(divUp((int)n, 256), 512);
    bboxKernel<<<blocks, 256, 0, s>>>(px, py, pz, bmin, bmax, n);
    padBBoxKernel<<<1, 1, 0, s>>>(bmin, bmax);
    CUDA_CHECK(cudaGetLastError());
}

__device__ int commonPrefix(const uint64_t* codes, int n, int i, int j) {
    if (j < 0 || j >= n) return -1;
    if (codes[i] == codes[j]) return 64 + __clz(i ^ j);
    return __clzll(codes[i] ^ codes[j]);
}

__global__ void initLeavesKernel(OctreeData tree, int numInt, int numLeaves) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numLeaves) return;
    int ni = numInt + i;
    tree.is_leaf[ni] = 1;
    tree.particle_start[ni] = i;
    tree.particle_count[ni] = 1;
    tree.left[ni] = -1;
    tree.right[ni] = -1;
    tree.atomic_visited[ni] = 0;
    tree.total_mass[ni] = 0.0f;
    tree.leftRange[ni] = i;
    tree.rightRange[ni] = i;
    tree.parentIdx[ni] = -1;
}

__global__ void initInternalKernel(OctreeData tree, int numInt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numInt) return;
    tree.is_leaf[i] = 0;
    tree.atomic_visited[i] = 0;
    tree.total_mass[i] = 0.0f;
    tree.com_x[i] = tree.com_y[i] = tree.com_z[i] = 0.0f;
    tree.particle_start[i] = -1;
    tree.particle_count[i] = 0;
    tree.parentIdx[i] = -1;
}

__global__ void buildTreeKernel(const uint64_t* __restrict__ codes, OctreeData tree, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;
    int numInt = n - 1;
    
    int d = (commonPrefix(codes, n, i, i + 1) - commonPrefix(codes, n, i, i - 1)) > 0 ? 1 : -1;
    int dmin = commonPrefix(codes, n, i, i - d);
    int lmax = 2;
    while (commonPrefix(codes, n, i, i + lmax * d) > dmin) lmax *= 2;
    
    int l = 0;
    for (int k = lmax / 2; k >= 1; k /= 2) {
        if (commonPrefix(codes, n, i, i + (l + k) * d) > dmin) l += k;
    }
    int j = i + l * d;
    int dn = commonPrefix(codes, n, i, j);
    int s = 0, div = 2;
    for (int k = (l + div - 1) / div; k >= 1; div *= 2, k = (l + div - 1) / div) {
        if (commonPrefix(codes, n, i, i + (s + k) * d) > dn) s += k;
    }
    
    int gam = i + s * d + min(d, 0);
    int lc = (min(i, j) == gam) ? numInt + gam : gam;
    int rc = (max(i, j) == gam + 1) ? numInt + gam + 1 : gam + 1;
    
    tree.left[i] = lc;
    tree.right[i] = rc;
    tree.parentIdx[lc] = i;
    tree.parentIdx[rc] = i;
    tree.leftRange[i] = min(i, j);
    tree.rightRange[i] = max(i, j);
}

void launchBuildTree(const uint64_t* codes, OctreeData& tree, uint32_t n, cudaStream_t s) {
    if (n <= 1) return;
    int numInt = n - 1;
    initLeavesKernel<<<divUp((int)n, 256), 256, 0, s>>>(tree, numInt, n);
    initInternalKernel<<<divUp(numInt, 256), 256, 0, s>>>(tree, numInt);
    CUDA_CHECK(cudaGetLastError());
    buildTreeKernel<<<divUp(numInt, 256), 256, 0, s>>>(codes, tree, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void nodePropsKernel(
    const float* __restrict__ px, const float* __restrict__ py, const float* __restrict__ pz,
    const float* __restrict__ mass, const uint32_t* __restrict__ si, 
    OctreeData tree, int numInt, int numLeaves
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numLeaves) return;
    
    int ni = numInt + i;
    uint32_t pi = si[i];
    float m = mass[pi];
    float x = px[pi], y = py[pi], z = pz[pi];
    
    tree.com_x[ni] = x * m; 
    tree.com_y[ni] = y * m; 
    tree.com_z[ni] = z * m;
    tree.total_mass[ni] = m;
    tree.bmin_x[ni] = tree.bmax_x[ni] = x;
    tree.bmin_y[ni] = tree.bmax_y[ni] = y;
    tree.bmin_z[ni] = tree.bmax_z[ni] = z;
    
    int cur = tree.parentIdx[ni];
    while (cur >= 0) {
        int old = atomicAdd(&tree.atomic_visited[cur], 1);
        if (old == 0) return;
        
        int lc = tree.left[cur];
        int rc = tree.right[cur];
        
        float tm = tree.total_mass[lc] + tree.total_mass[rc];
        tree.total_mass[cur] = tm;
        tree.com_x[cur] = tree.com_x[lc] + tree.com_x[rc];
        tree.com_y[cur] = tree.com_y[lc] + tree.com_y[rc];
        tree.com_z[cur] = tree.com_z[lc] + tree.com_z[rc];
        
        tree.bmin_x[cur] = fminf(tree.bmin_x[lc], tree.bmin_x[rc]);
        tree.bmin_y[cur] = fminf(tree.bmin_y[lc], tree.bmin_y[rc]);
        tree.bmin_z[cur] = fminf(tree.bmin_z[lc], tree.bmin_z[rc]);
        
        tree.bmax_x[cur] = fmaxf(tree.bmax_x[lc], tree.bmax_x[rc]);
        tree.bmax_y[cur] = fmaxf(tree.bmax_y[lc], tree.bmax_y[rc]);
        tree.bmax_z[cur] = fmaxf(tree.bmax_z[lc], tree.bmax_z[rc]);
        
        tree.particle_start[cur] = tree.leftRange[cur];
        tree.particle_count[cur] = tree.rightRange[cur] - tree.leftRange[cur] + 1;
        
        cur = tree.parentIdx[cur];
    }
}

void launchNodeProperties(const float* px, const float* py, const float* pz, 
                          const float* mass, const uint32_t* si, 
                          OctreeData& tree, uint32_t n, cudaStream_t s) {
    if (n <= 1) return;
    nodePropsKernel<<<divUp((int)n, 256), 256, 0, s>>>(
        px, py, pz, mass, si, tree, tree.numInternal, n);
    CUDA_CHECK(cudaGetLastError());
}