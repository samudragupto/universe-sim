#include "physics.cuh"
#include "cuda_utils.h"

__global__ void brK(const float* px, const float* py, const float* pz, const float* mass, float* ax, float* ay, float* az, const uint8_t* al, float G, float e2, uint32_t n) {
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n||!al[i]) return;
    float xi=px[i], yi=py[i], zi=pz[i], fx=0, fy=0, fz=0;
    for(uint32_t j=0; j<n; j++) { 
        if(j==i||!al[j]) continue; 
        float dx=px[j]-xi, dy=py[j]-yi, dz=pz[j]-zi, d2=dx*dx+dy*dy+dz*dz+e2, iv=rsqrtf(d2), f=G*mass[j]*iv*iv*iv; 
        fx+=f*dx; fy+=f*dy; fz+=f*dz; 
    }
    ax[i]=fx; ay[i]=fy; az[i]=fz; 
}

void launchBruteForce(ParticleDeviceData& d, float G, float eps, uint32_t n, cudaStream_t s) { 
    brK<<<divUp(n,256),256,0,s>>>(d.pos_x,d.pos_y,d.pos_z,d.mass,d.acc_x,d.acc_y,d.acc_z,d.alive,G,eps*eps,n); 
    CUDA_CHECK(cudaGetLastError()); 
}

__global__ void bhK(
    const float* __restrict__ px, const float* __restrict__ py, const float* __restrict__ pz,
    float* __restrict__ ax, float* __restrict__ ay, float* __restrict__ az,
    const uint8_t* __restrict__ alive, 
    OctreeData t, 
    const uint32_t* __restrict__ si, float G, float e2, float th2, int nI, uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i >= n || !alive[i]) return;
    
    float xi = px[i], yi = py[i], zi = pz[i];
    float fx = 0, fy = 0, fz = 0; 
    
    int stk[64];
    int sp = 0; 
    stk[sp++] = 0;
    
    while (sp > 0) { 
        int ni = stk[--sp]; 
        if (ni < 0 || ni >= (int)(nI + n)) continue; 
        
        float tm = t.total_mass[ni]; 
        if (tm <= 0.0f) continue;
        
        float ivM = 1.0f / tm;
        float cx = t.com_x[ni] * ivM;
        float cy = t.com_y[ni] * ivM;
        float cz = t.com_z[ni] * ivM;
        
        float dx = cx - xi, dy = cy - yi, dz = cz - zi;
        float d2 = dx*dx + dy*dy + dz*dz + e2;
        
        if (t.is_leaf[ni]) { 
            if (t.particle_count[ni] == 1 && si[t.particle_start[ni]] == i) continue; 
            float iv = rsqrtf(d2); 
            float f = G * tm * iv * iv * iv; 
            fx += f*dx; fy += f*dy; fz += f*dz; 
            continue; 
        }
        
        float sx = t.bmax_x[ni] - t.bmin_x[ni];
        float sy = t.bmax_y[ni] - t.bmin_y[ni];
        float sz = t.bmax_z[ni] - t.bmin_z[ni];
        float s2 = fmaxf(sx, fmaxf(sy, sz)); 
        s2 *= s2;
        
        if (s2 < th2 * d2) { 
            float iv = rsqrtf(d2); 
            float f = G * tm * iv * iv * iv; 
            fx += f*dx; fy += f*dy; fz += f*dz; 
        } else if (sp + 2 <= 64) { 
            stk[sp++] = t.left[ni]; 
            stk[sp++] = t.right[ni]; 
        } 
    }
    ax[i] = fx; ay[i] = fy; az[i] = fz; 
}

void launchBarnesHut(ParticleDeviceData& d, const OctreeData& t, const uint32_t* si, float G, float eps, float th, uint32_t n, cudaStream_t s) { 
    bhK<<<divUp(n,256),256,0,s>>>(d.pos_x, d.pos_y, d.pos_z, d.acc_x, d.acc_y, d.acc_z, d.alive, t, si, G, eps*eps, th*th, t.numInternal, n); 
    CUDA_CHECK(cudaGetLastError()); 
}

__global__ void intK(float* px, float* py, float* pz, float* vx, float* vy, float* vz, const float* ax, const float* ay, const float* az, const uint8_t* al, float* ag, float dt, uint32_t n) {
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n||!al[i]) return; vx[i]+=ax[i]*dt; vy[i]+=ay[i]*dt; vz[i]+=az[i]*dt; px[i]+=vx[i]*dt; py[i]+=vy[i]*dt; pz[i]+=vz[i]*dt; ag[i]+=dt; }
void launchIntegrate(ParticleDeviceData& d, float dt, uint32_t n, cudaStream_t s) { intK<<<divUp(n,256),256,0,s>>>(d.pos_x,d.pos_y,d.pos_z,d.vel_x,d.vel_y,d.vel_z,d.acc_x,d.acc_y,d.acc_z,d.alive,d.age,dt,n); CUDA_CHECK(cudaGetLastError()); }

__global__ void colK(float* px, float* py, float* pz, float* vx, float* vy, float* vz, float* m, uint8_t* typ, uint8_t* al, float md2, uint32_t n) {
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n||!al[i]||typ[i]!=2) return; float bx=px[i], by=py[i], bz=pz[i], bm=m[i], rs2=bm*bm*1e-4f;
    for(uint32_t j=0; j<n; j++) { if(j==i||!al[j]) continue; float dx=px[j]-bx, dy=py[j]-by, dz=pz[j]-bz; if(dx*dx+dy*dy+dz*dz<rs2) { float mj=m[j], nm=bm+mj; vx[i]=(vx[i]*bm+vx[j]*mj)/nm; vy[i]=(vy[i]*bm+vy[j]*mj)/nm; vz[i]=(vz[i]*bm+vz[j]*mj)/nm; m[i]=nm; bm=nm; al[j]=0; } } }
void launchCollisions(ParticleDeviceData& d, float md, uint32_t n, cudaStream_t s) { colK<<<divUp(n,256),256,0,s>>>(d.pos_x,d.pos_y,d.pos_z,d.vel_x,d.vel_y,d.vel_z,d.mass,d.type,d.alive,md*md,n); CUDA_CHECK(cudaGetLastError()); }