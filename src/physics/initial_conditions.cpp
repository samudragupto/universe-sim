#include "initial_conditions.h"
#include "cuda/cuda_utils.h"
#include <cmath>
#include <random>
#include <vector>

Scenario scenarioFromString(const std::string& s) {
    if (s == "big_bang") return Scenario::BIG_BANG;
    if (s == "galaxy_collision") return Scenario::GALAXY_COLLISION;
    if (s == "protogalactic_cloud") return Scenario::PROTOGALACTIC_CLOUD;
    if (s == "solar_system") return Scenario::SOLAR_SYSTEM;
    return Scenario::RANDOM_SPHERE;
}

void InitialConditions::generate(ParticleSystem& sys, Scenario sc, uint32_t n) {
    sys.allocate(n);
    switch (sc) {
        case Scenario::BIG_BANG: bigBang(sys, n); break;
        case Scenario::GALAXY_COLLISION: galaxyCollision(sys, n); break;
        case Scenario::PROTOGALACTIC_CLOUD: protoCloud(sys, n); break;
        case Scenario::SOLAR_SYSTEM: solarSystem(sys, n); break;
        default: randomSphere(sys, n); break;
    }
}

static void upload(ParticleSystem& sys, uint32_t n,
    std::vector<float>& px, std::vector<float>& py, std::vector<float>& pz,
    std::vector<float>& vx, std::vector<float>& vy, std::vector<float>& vz,
    std::vector<float>& mass, std::vector<float>& temp, std::vector<float>& lum,
    std::vector<uint8_t>& type, std::vector<uint8_t>& alive)
{
    auto& d = sys.deviceData();
    CUDA_CHECK(cudaMemcpy(d.pos_x, px.data(), n*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.pos_y, py.data(), n*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.pos_z, pz.data(), n*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vel_x, vx.data(), n*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vel_y, vy.data(), n*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vel_z, vz.data(), n*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.mass, mass.data(), n*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.temperature, temp.data(), n*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.luminosity, lum.data(), n*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.type, type.data(), n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.alive, alive.data(), n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d.acc_x, 0, n*4));
    CUDA_CHECK(cudaMemset(d.acc_y, 0, n*4));
    CUDA_CHECK(cudaMemset(d.acc_z, 0, n*4));
    std::vector<float> rad(n, 0.01f), ages(n, 0.0f);
    CUDA_CHECK(cudaMemcpy(d.radius, rad.data(), n*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.age, ages.data(), n*4, cudaMemcpyHostToDevice));
}

void InitialConditions::bigBang(ParticleSystem& sys, uint32_t n) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(0,1);
    std::normal_distribution<float> g(0,1);
    std::vector<float> px(n),py(n),pz(n),vx(n),vy(n),vz(n),mass(n),temp(n),lum(n);
    std::vector<uint8_t> type(n,0),alive(n,1);
    for (uint32_t i = 0; i < n; i++) {
        float r = 0.5f * std::cbrt(u(rng));
        float t = 6.283f * u(rng), p = std::acos(2*u(rng)-1);
        px[i] = r*sinf(p)*cosf(t) + g(rng)*0.01f;
        py[i] = r*sinf(p)*sinf(t) + g(rng)*0.01f;
        pz[i] = r*cosf(p) + g(rng)*0.01f;
        float d = sqrtf(px[i]*px[i]+py[i]*py[i]+pz[i]*pz[i]);
        float h = 0.5f;
        vx[i] = h*px[i]; vy[i] = h*py[i]; vz[i] = h*pz[i];
        mass[i] = 1.0f; temp[i] = 8000+g(rng)*2000; lum[i] = 0.5f+u(rng)*1.5f;
    }
    upload(sys, n, px, py, pz, vx, vy, vz, mass, temp, lum, type, alive);
}

void InitialConditions::galaxyCollision(ParticleSystem& sys, uint32_t n) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> u(0,1);
    std::normal_distribution<float> g(0,1);
    uint32_t half = n/2;
    std::vector<float> px(n),py(n),pz(n),vx(n),vy(n),vz(n),mass(n),temp(n),lum(n);
    std::vector<uint8_t> type(n,0),alive(n,1);
    auto galaxy = [&](uint32_t s, uint32_t cnt, float cx, float cy, float cz, float cvx, float cvy, float cvz, float R) {
        uint32_t bulge = cnt/5;
        for (uint32_t i = 0; i < bulge; i++) {
            uint32_t idx = s+i;
            float r = R*0.2f*std::cbrt(u(rng));
            float t=6.283f*u(rng), p=acosf(2*u(rng)-1);
            px[idx]=cx+r*sinf(p)*cosf(t); py[idx]=cy+r*sinf(p)*sinf(t); pz[idx]=cz+r*cosf(p);
            vx[idx]=cvx+g(rng)*0.2f; vy[idx]=cvy+g(rng)*0.2f; vz[idx]=cvz+g(rng)*0.2f;
            mass[idx]=1.0f+u(rng)*2; temp[idx]=5000+u(rng)*5000; lum[idx]=0.5f+u(rng);
        }
        for (uint32_t i = bulge; i < cnt; i++) {
            uint32_t idx = s+i;
            float r = R*std::sqrt(u(rng));
            float t = 6.283f*u(rng);
            px[idx]=cx+r*cosf(t); py[idx]=cy+r*sinf(t); pz[idx]=cz+g(rng)*0.3f;
            float orb = std::sqrt(bulge*1.0f/(r+0.5f));
            vx[idx]=cvx-orb*sinf(t); vy[idx]=cvy+orb*cosf(t); vz[idx]=cvz+g(rng)*0.01f;
            mass[idx]=0.5f+u(rng); temp[idx]=3000+u(rng)*15000; lum[idx]=0.3f+u(rng)*1.5f;
            type[idx]=(u(rng)<0.1f)?3:0;
        }
    };
    galaxy(0, half, -20, 0, 0, 0.5f, 0.2f, 0, 12);
    galaxy(half, n-half, 20, 0, 0, -0.5f, -0.2f, 0, 12);
    upload(sys, n, px, py, pz, vx, vy, vz, mass, temp, lum, type, alive);
}

void InitialConditions::protoCloud(ParticleSystem& sys, uint32_t n) {
    std::mt19937 rng(777);
    std::uniform_real_distribution<float> u(0,1);
    std::normal_distribution<float> g(0,1);
    std::vector<float> px(n),py(n),pz(n),vx(n),vy(n),vz(n),mass(n),temp(n),lum(n);
    std::vector<uint8_t> type(n,0),alive(n,1);
    for (uint32_t i = 0; i < n; i++) {
        float r = 30*std::cbrt(u(rng))*(1+g(rng)*0.05f);
        float t=6.283f*u(rng), p=acosf(2*u(rng)-1);
        px[i]=r*sinf(p)*cosf(t); py[i]=r*sinf(p)*sinf(t); pz[i]=r*cosf(p);
        float ang = 0.03f;
        vx[i]=-ang*py[i]+g(rng)*0.05f; vy[i]=ang*px[i]+g(rng)*0.05f; vz[i]=g(rng)*0.02f;
        mass[i]=0.5f+u(rng)*1.5f; temp[i]=5000+u(rng)*10000; lum[i]=0.5f+u(rng);
        type[i]=(u(rng)<0.3f)?3:0;
    }
    upload(sys, n, px, py, pz, vx, vy, vz, mass, temp, lum, type, alive);
}

void InitialConditions::solarSystem(ParticleSystem& sys, uint32_t n) {
    std::mt19937 rng(314);
    std::uniform_real_distribution<float> u(0,1);
    std::normal_distribution<float> g(0,1);
    std::vector<float> px(n),py(n),pz(n),vx(n),vy(n),vz(n),mass(n),temp(n),lum(n);
    std::vector<uint8_t> type(n,0),alive(n,1);
    px[0]=py[0]=pz[0]=vx[0]=vy[0]=vz[0]=0;
    mass[0]=100000; temp[0]=5778; lum[0]=5; type[0]=0;
    for (uint32_t i = 1; i < n; i++) {
        float r = 1+u(rng)*40;
        float t = 6.283f*u(rng);
        px[i]=r*cosf(t); py[i]=r*sinf(t); pz[i]=g(rng)*0.1f*r;
        float orb = sqrtf(mass[0]/r);
        vx[i]=-orb*sinf(t); vy[i]=orb*cosf(t); vz[i]=0;
        mass[i]=0.001f+u(rng)*0.05f; temp[i]=500+u(rng)*2000; lum[i]=0.1f+u(rng)*0.3f;
        type[i]=3;
    }
    upload(sys, n, px, py, pz, vx, vy, vz, mass, temp, lum, type, alive);
}

void InitialConditions::randomSphere(ParticleSystem& sys, uint32_t n) {
    sys.initRandom(12345ULL);
}