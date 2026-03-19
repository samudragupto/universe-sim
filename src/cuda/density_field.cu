#include "density_field.cuh"
#include "cuda_utils.h"

void densityFieldAllocate(DensityField& df, int res) {
    df.resX = df.resY = df.resZ = res;
    df.allocated = true;
    size_t totalCells = res * res * res;
    CUDA_CHECK(cudaMalloc(&df.data, totalCells * sizeof(float)));
    CUDA_CHECK(cudaMemset(df.data, 0, totalCells * sizeof(float)));
}

void densityFieldDeallocate(DensityField& df) {
    if (df.allocated && df.data) cudaFree(df.data);
    df.data = nullptr;
    df.allocated = false;
}

__global__ void clearFieldKernel(float* data, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    data[i] = 0.0f;
}

__global__ void scatterKernel(
    const float* __restrict__ px, const float* __restrict__ py, const float* __restrict__ pz,
    const float* __restrict__ mass, const uint8_t* __restrict__ type, const uint8_t* __restrict__ alive,
    float* __restrict__ field,
    float bminX, float bminY, float bminZ,
    float invSX, float invSY, float invSZ,
    int resX, int resY, int resZ, uint32_t count
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count || !alive[i]) return;
    if (type[i] != 3 && type[i] != 0) return;

    float nx = (px[i] - bminX) * invSX;
    float ny = (py[i] - bminY) * invSY;
    float nz = (pz[i] - bminZ) * invSZ;

    int ix = (int)(nx * (resX - 1));
    int iy = (int)(ny * (resY - 1));
    int iz = (int)(nz * (resZ - 1));

    if (ix < 0 || ix >= resX || iy < 0 || iy >= resY || iz < 0 || iz >= resZ) return;

    float contribution = mass[i] * 0.1f;
    if (type[i] == 3) contribution *= 3.0f;

    int idx = iz * resY * resX + iy * resX + ix;
    atomicAdd(&field[idx], contribution);
}

void launchScatterDensity(const ParticleDeviceData& p, DensityField& df,
                          const float* bboxMin, const float* bboxMax,
                          uint32_t count, cudaStream_t stream) {
    int total = df.resX * df.resY * df.resZ;
    clearFieldKernel<<<divUp(total, 256), 256, 0, stream>>>(df.data, total);

    float hMin[3], hMax[3];
    CUDA_CHECK(cudaMemcpyAsync(hMin, bboxMin, 3 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(hMax, bboxMax, 3 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float sx = hMax[0] - hMin[0], sy = hMax[1] - hMin[1], sz = hMax[2] - hMin[2];
    float invSX = (sx > 1e-6f) ? 1.0f / sx : 0.0f;
    float invSY = (sy > 1e-6f) ? 1.0f / sy : 0.0f;
    float invSZ = (sz > 1e-6f) ? 1.0f / sz : 0.0f;

    df.minX = hMin[0]; df.minY = hMin[1]; df.minZ = hMin[2];
    df.maxX = hMax[0]; df.maxY = hMax[1]; df.maxZ = hMax[2];

    scatterKernel<<<divUp(count, 256), 256, 0, stream>>>(
        p.pos_x, p.pos_y, p.pos_z, p.mass, p.type, p.alive,
        df.data, hMin[0], hMin[1], hMin[2], invSX, invSY, invSZ,
        df.resX, df.resY, df.resZ, count);
    CUDA_CHECK(cudaGetLastError());
}