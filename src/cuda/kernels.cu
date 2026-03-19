#include "kernels.cuh"
#include "cuda_utils.h"
#include <curand_kernel.h>

__device__ void bbToRGB(float tK, float& r, float& g, float& b) {
    float t = tK / 100.0f;
    if(t<=66.0f) { r=255; g=99.47f*logf(fmaxf(t,1.0f))-161.1f; b=(t<=19.0f)?0:138.5f*logf(fmaxf(t-10.0f,1.0f))-305.0f; }
    else { r=329.7f*powf(fmaxf(t-60.0f,1.0f),-0.133f); g=288.1f*powf(fmaxf(t-60.0f,1.0f),-0.075f); b=255; }
    r=fminf(fmaxf(r,0.0f),255.0f)/255.0f; g=fminf(fmaxf(g,0.0f),255.0f)/255.0f; b=fminf(fmaxf(b,0.0f),255.0f)/255.0f;
}

__global__ void initRandK(float* px, float* py, float* pz, float* vx, float* vy, float* vz, float* ax, float* ay, float* az,
                          float* m, float* r, float* tmp, float* lum, float* age, uint8_t* typ, uint8_t* al, uint32_t n, unsigned long long sd) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; if(i>=n) return;
    curandState st; curand_init(sd, i, 0, &st);
    float rad = curand_uniform(&st)*50.0f, th=curand_uniform(&st)*6.283f, ph=acosf(2.0f*curand_uniform(&st)-1.0f);
    px[i]=rad*sinf(ph)*cosf(th); py[i]=rad*sinf(ph)*sinf(th); pz[i]=rad*cosf(ph);
    vx[i]=vy[i]=vz[i]=ax[i]=ay[i]=az[i]=0; m[i]=1; r[i]=0.01f; tmp[i]=3000.0f+curand_uniform(&st)*27000.0f;
    lum[i]=1; age[i]=0; typ[i]=PARTICLE_STAR; al[i]=1;
}

__global__ void compColK(const float* px, const float* py, const float* pz, const float* vx, const float* vy, const float* vz,
                         const float* tmp, const float* lum, const float* m, const float* r, const uint8_t* typ, const uint8_t* al, const float* age, RenderVertex* out, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; if(i>=n) return;
    RenderVertex v; v.px=px[i]; v.py=py[i]; v.pz=pz[i];
    if(!al[i]) { v.ca=0; v.size=0; out[i]=v; return; }
    float sp = sqrtf(vx[i]*vx[i]+vy[i]*vy[i]+vz[i]*vz[i]);
    switch(typ[i]) {
        case PARTICLE_STAR: { float cr,cg,cb; bbToRGB(tmp[i],cr,cg,cb); float l=fminf(lum[i],10.f), vb=1.0f+sp*0.005f;
            v.cr=cr*l*vb; v.cg=cg*l*vb; v.cb=cb*l*vb; v.ca=fminf(1.f,0.5f+l*.15f); v.size=0.02f+l*.01f; break; }
        case PARTICLE_DM: v.cr=0.4f; v.cg=0.2f; v.cb=0.8f; v.ca=0.03f; v.size=0.01f; break;
        case PARTICLE_BH: { float rg=sinf(age[i]*50.f)*.5f+.5f; v.cr=rg*.8f; v.cg=rg*.4f; v.cb=rg*.1f; v.ca=1; v.size=fmaxf(.1f,m[i]*.001f); break; }
        case PARTICLE_GAS: { float nm=fminf(tmp[i]/10000.f,1.f); v.cr=1; v.cg=.3f+nm*.3f; v.cb=.1f+nm*.4f; v.ca=fminf(.3f,.1f+m[i]*.005f); v.size=0.03f; break; }
        case PARTICLE_NS: { float pl=sinf(age[i]*200.f)*.5f+.5f; v.cr=0; v.cg=.7f+pl*.3f; v.cb=.8f+pl*.2f; v.ca=.9f; v.size=0.025f; break; }
        default: v.cr=v.cg=v.cb=v.ca=1; v.size=0.02f;
    } out[i]=v;
}

__global__ void clrAccK(float* ax, float* ay, float* az, uint32_t n) { uint32_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) ax[i]=ay[i]=az[i]=0; }
void launchInitRandom(ParticleDeviceData& d, unsigned long long sd, cudaStream_t s) { initRandK<<<divUp(d.count,256),256,0,s>>>(d.pos_x,d.pos_y,d.pos_z,d.vel_x,d.vel_y,d.vel_z,d.acc_x,d.acc_y,d.acc_z,d.mass,d.radius,d.temperature,d.luminosity,d.age,d.type,d.alive,d.count,sd); CUDA_CHECK(cudaGetLastError()); }
void launchComputeColors(ParticleDeviceData& d, RenderVertex* buf, uint32_t n, cudaStream_t s) { compColK<<<divUp(n,256),256,0,s>>>(d.pos_x,d.pos_y,d.pos_z,d.vel_x,d.vel_y,d.vel_z,d.temperature,d.luminosity,d.mass,d.radius,d.type,d.alive,d.age,buf,n); CUDA_CHECK(cudaGetLastError()); }
void launchClearAccelerations(ParticleDeviceData& d, cudaStream_t s) { clrAccK<<<divUp(d.count,256),256,0,s>>>(d.acc_x,d.acc_y,d.acc_z,d.count); CUDA_CHECK(cudaGetLastError()); }