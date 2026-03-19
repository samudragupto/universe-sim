#include "physics.cuh"
#include "cuda_utils.h"

__global__ void bruteForceKernel(
    const float* px, const float* py, const float* pz,
    const float* mass,
    float* ax, float* ay, float* az,
    const uint8_t* alive,
    float G, float eps2, uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !alive[i]) return;

    float xi = px[i], yi = py[i], zi = pz[i];
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    for (uint32_t j = 0; j < n; j++) {
        if (j == i || !alive[j]) continue;
        float dx = px[j] - xi;
        float dy = py[j] - yi;
        float dz = pz[j] - zi;
        float d2 = dx * dx + dy * dy + dz * dz + eps2;
        float inv = rsqrtf(d2);
        float f = G * mass[j] * inv * inv * inv;
        fx += f * dx;
        fy += f * dy;
        fz += f * dz;
    }

    ax[i] = fx;
    ay[i] = fy;
    az[i] = fz;
}

void launchBruteForce(ParticleDeviceData& d, float G, float eps, uint32_t n, cudaStream_t s) {
    bruteForceKernel<<<divUp(n, 256), 256, 0, s>>>(
        d.pos_x, d.pos_y, d.pos_z,
        d.mass,
        d.acc_x, d.acc_y, d.acc_z,
        d.alive,
        G, eps * eps, n
    );
    CUDA_CHECK(cudaGetLastError());
}

#define BH_STACK_SIZE 48

__global__ void barnesHutKernel(
    const float* __restrict__ px,
    const float* __restrict__ py,
    const float* __restrict__ pz,
    float* __restrict__ ax,
    float* __restrict__ ay,
    float* __restrict__ az,
    const uint8_t* __restrict__ alive,
    OctreeData tree,
    const uint32_t* __restrict__ sortedIndices,
    float G,
    float eps2,
    float theta2,
    int numInternal,
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !alive[i]) return;

    float xi = px[i];
    float yi = py[i];
    float zi = pz[i];

    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    int stack[BH_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int ni = stack[--sp];
        if (ni < 0 || ni >= (int)(numInternal + n)) continue;

        float nodeMass = tree.total_mass[ni];
        if (nodeMass <= 0.0f) continue;

        float invMass = 1.0f / nodeMass;
        float cx = tree.com_x[ni] * invMass;
        float cy = tree.com_y[ni] * invMass;
        float cz = tree.com_z[ni] * invMass;

        float dx = cx - xi;
        float dy = cy - yi;
        float dz = cz - zi;

        float d2 = dx * dx + dy * dy + dz * dz + eps2;

        bool leafLike = tree.is_leaf[ni] || tree.particle_count[ni] <= 32;

        if (leafLike) {
            if (tree.particle_count[ni] == 1) {
                uint32_t pidx = sortedIndices[tree.particle_start[ni]];
                if (pidx == i) continue;
            }
            float inv = rsqrtf(d2);
            float f = G * nodeMass * inv * inv * inv;
            fx += f * dx;
            fy += f * dy;
            fz += f * dz;
            continue;
        }

        float sx = tree.bmax_x[ni] - tree.bmin_x[ni];
        float sy = tree.bmax_y[ni] - tree.bmin_y[ni];
        float sz = tree.bmax_z[ni] - tree.bmin_z[ni];
        float s = fmaxf(sx, fmaxf(sy, sz));
        float size2 = s * s;

        if (size2 < theta2 * d2) {
            float inv = rsqrtf(d2);
            float f = G * nodeMass * inv * inv * inv;
            fx += f * dx;
            fy += f * dy;
            fz += f * dz;
        } else {
            if (sp + 2 <= BH_STACK_SIZE) {
                stack[sp++] = tree.left[ni];
                stack[sp++] = tree.right[ni];
            }
        }
    }

    ax[i] = fx;
    ay[i] = fy;
    az[i] = fz;
}

void launchBarnesHut(ParticleDeviceData& d, const OctreeData& t, const uint32_t* si, float G, float eps, float theta, uint32_t n, cudaStream_t s) {
    barnesHutKernel<<<divUp(n, 256), 256, 0, s>>>(
        d.pos_x,
        d.pos_y,
        d.pos_z,
        d.acc_x,
        d.acc_y,
        d.acc_z,
        d.alive,
        t,
        si,
        G,
        eps * eps,
        theta * theta,
        t.numInternal,
        n
    );
    CUDA_CHECK(cudaGetLastError());
}

__global__ void integrateKernel(
    float* px, float* py, float* pz,
    float* vx, float* vy, float* vz,
    const float* ax, const float* ay, const float* az,
    const uint8_t* alive,
    float* age,
    float dt,
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !alive[i]) return;

    vx[i] += ax[i] * dt;
    vy[i] += ay[i] * dt;
    vz[i] += az[i] * dt;

    px[i] += vx[i] * dt;
    py[i] += vy[i] * dt;
    pz[i] += vz[i] * dt;

    age[i] += dt;
}

void launchIntegrate(ParticleDeviceData& d, float dt, uint32_t n, cudaStream_t s) {
    integrateKernel<<<divUp(n, 256), 256, 0, s>>>(
        d.pos_x, d.pos_y, d.pos_z,
        d.vel_x, d.vel_y, d.vel_z,
        d.acc_x, d.acc_y, d.acc_z,
        d.alive,
        d.age,
        dt,
        n
    );
    CUDA_CHECK(cudaGetLastError());
}

__global__ void collisionKernel(
    float* px, float* py, float* pz,
    float* vx, float* vy, float* vz,
    float* m,
    uint8_t* typ,
    uint8_t* al,
    float md2,
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !al[i] || typ[i] != 2) return;

    float bx = px[i], by = py[i], bz = pz[i];
    float bm = m[i];
    float rs2 = bm * bm * 1e-4f;

    for (uint32_t j = 0; j < n; j++) {
        if (j == i || !al[j]) continue;
        float dx = px[j] - bx;
        float dy = py[j] - by;
        float dz = pz[j] - bz;
        if (dx * dx + dy * dy + dz * dz < rs2) {
            float mj = m[j];
            float nm = bm + mj;
            vx[i] = (vx[i] * bm + vx[j] * mj) / nm;
            vy[i] = (vy[i] * bm + vy[j] * mj) / nm;
            vz[i] = (vz[i] * bm + vz[j] * mj) / nm;
            m[i] = nm;
            bm = nm;
            al[j] = 0;
        }
    }
}

void launchCollisions(ParticleDeviceData& d, float mergeDist, uint32_t n, cudaStream_t s) {
    collisionKernel<<<divUp(n, 256), 256, 0, s>>>(
        d.pos_x, d.pos_y, d.pos_z,
        d.vel_x, d.vel_y, d.vel_z,
        d.mass,
        d.type,
        d.alive,
        mergeDist * mergeDist,
        n
    );
    CUDA_CHECK(cudaGetLastError());
}