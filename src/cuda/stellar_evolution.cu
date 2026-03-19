#include "stellar_evolution.cuh"
#include "cuda_utils.h"
__global__ void evK(float* m, float* tmp, float* lum, float* age, float* rad, uint8_t* typ, uint8_t* al, float dt, uint32_t n) {
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n||!al[i]) return; uint8_t t=typ[i]; float mass=m[i], a=age[i];
    if(t==PARTICLE_GAS && mass>50.f) { typ[i]=PARTICLE_STAR; tmp[i]=3000.f+mass*40.f; lum[i]=.3f; }
    if(t==PARTICLE_STAR) { float lf=10000.f/fmaxf(mass*mass*mass,0.001f); if(a<lf) { if(mass>16.f){tmp[i]=30000.f+mass*500.f;lum[i]=mass*mass*0.01f;}else if(mass>2.f){tmp[i]=10000.f+mass*1000.f;lum[i]=mass*.5f;}else{tmp[i]=4000.f+mass*1000.f;lum[i]=mass*mass;} } else if(a<lf*1.3f) { tmp[i]=3500.f; lum[i]=fminf(lum[i]*1.005f,50.f); } else { if(mass>25.f){typ[i]=PARTICLE_BH;m[i]=mass*.3f;lum[i]=0;}else if(mass>8.f){typ[i]=PARTICLE_NS;m[i]=fminf(mass*.1f,3.f);lum[i]=2.f;}else{lum[i]=.01f;tmp[i]=15000.f;} } }
    if(t==PARTICLE_NS) lum[i]=1.f+sinf(a*100.f)*.5f; }
void launchStellarEvolution(ParticleDeviceData& d, float dt, uint32_t n, cudaStream_t s) { evK<<<divUp(n,256),256,0,s>>>(d.mass,d.temperature,d.luminosity,d.age,d.radius,d.type,d.alive,dt,n); CUDA_CHECK(cudaGetLastError()); }