#include "diagnostics.cuh"
#include "cuda_utils.h"

__global__ void diagK(const float* px, const float* py, const float* pz, const float* vx, const float* vy, const float* vz, const float* ax, const float* ay, const float* az, const float* m, const uint8_t* typ, const uint8_t* al, float* r, uint32_t n) {
    __shared__ float sKE[256], sCX[256], sCY[256], sCZ[256], sM[256], sMV[256], sMA[256];
    __shared__ uint32_t sAl[256], sSt[256], sGs[256], sBH[256], sDM[256], sNS[256];
    
    uint32_t t = threadIdx.x;
    uint32_t i = blockIdx.x * blockDim.x + t;
    
    float ke=0, cx=0, cy=0, cz=0, ms=0, mv=0, ma=0;
    uint32_t a=0, s=0, g=0, b=0, d=0, ns=0;
    
    if (i < n && al[i]) {
        float v2 = vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
        float a2 = ax[i]*ax[i] + ay[i]*ay[i] + az[i]*az[i];
        ke = 0.5f * m[i] * v2;
        ms = m[i];
        cx = px[i]*ms; cy = py[i]*ms; cz = pz[i]*ms;
        mv = sqrtf(v2); ma = sqrtf(a2);
        a = 1;
        switch(typ[i]) {
            case 0: s=1; break;
            case 1: d=1; break;
            case 2: b=1; break;
            case 3: g=1; break;
            case 4: ns=1; break;
        }
    }
    
    sKE[t]=ke; sCX[t]=cx; sCY[t]=cy; sCZ[t]=cz; sM[t]=ms; sMV[t]=mv; sMA[t]=ma;
    sAl[t]=a; sSt[t]=s; sGs[t]=g; sBH[t]=b; sDM[t]=d; sNS[t]=ns;
    __syncthreads();
    
    for (int k = 128; k > 0; k >>= 1) {
        if (t < k) {
            sKE[t] += sKE[t+k]; sCX[t] += sCX[t+k]; sCY[t] += sCY[t+k]; sCZ[t] += sCZ[t+k]; sM[t] += sM[t+k];
            sMV[t] = fmaxf(sMV[t], sMV[t+k]); sMA[t] = fmaxf(sMA[t], sMA[t+k]);
            sAl[t] += sAl[t+k]; sSt[t] += sSt[t+k]; sGs[t] += sGs[t+k]; sBH[t] += sBH[t+k]; sDM[t] += sDM[t+k]; sNS[t] += sNS[t+k];
        }
        __syncthreads();
    }
    
    if (t == 0) {
        atomicAdd(&r[0], sKE[0]); atomicAdd(&r[1], sCX[0]); atomicAdd(&r[2], sCY[0]); atomicAdd(&r[3], sCZ[0]); atomicAdd(&r[4], sM[0]);
        atomicAdd((int*)&r[5], (int)sAl[0]); atomicAdd((int*)&r[6], (int)sSt[0]); atomicAdd((int*)&r[7], (int)sGs[0]);
        atomicAdd((int*)&r[8], (int)sBH[0]); atomicAdd((int*)&r[9], (int)sDM[0]); atomicAdd((int*)&r[10], (int)sNS[0]);
        
        int oV = __float_as_int(r[11]), nV = __float_as_int(sMV[0]);
        while(nV > oV) { int a = oV; oV = atomicCAS((int*)&r[11], a, nV); if(oV == a) break; }
        
        int oA = __float_as_int(r[12]), nA = __float_as_int(sMA[0]);
        while(nA > oA) { int a = oA; oA = atomicCAS((int*)&r[12], a, nA); if(oA == a) break; }
    }
}

void launchComputeDiagnostics(const ParticleDeviceData& d, SimDiagnostics* hr, float G, uint32_t n, cudaStream_t s) {
    float* dr;
    CUDA_CHECK(cudaMalloc(&dr, 16*4));
    CUDA_CHECK(cudaMemsetAsync(dr, 0, 16*4, s));
    
    diagK<<<divUp(n, 256), 256, 0, s>>>(d.pos_x, d.pos_y, d.pos_z, d.vel_x, d.vel_y, d.vel_z, d.acc_x, d.acc_y, d.acc_z, d.mass, d.type, d.alive, dr, n);
    
    float h[16];
    CUDA_CHECK(cudaMemcpyAsync(h, dr, 16*4, cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    
    hr->totalKineticEnergy = h[0];
    float m = h[4];
    if (m > 0) { hr->comX = h[1]/m; hr->comY = h[2]/m; hr->comZ = h[3]/m; }
    
    hr->aliveCount = *(uint32_t*)&h[5];
    hr->starCount = *(uint32_t*)&h[6];
    hr->gasCount = *(uint32_t*)&h[7];
    hr->bhCount = *(uint32_t*)&h[8];
    hr->dmCount = *(uint32_t*)&h[9];
    hr->nsCount = *(uint32_t*)&h[10];
    hr->maxVelocity = h[11];
    hr->maxAcceleration = h[12];
    hr->totalPotentialEnergy = 0.0f;
    hr->totalEnergy = hr->totalKineticEnergy + hr->totalPotentialEnergy;
    
    cudaFree(dr);
}