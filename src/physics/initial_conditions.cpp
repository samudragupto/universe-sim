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

static void uploadToGPU(
    ParticleSystem& sys,
    uint32_t n,
    std::vector<float>& px, std::vector<float>& py, std::vector<float>& pz,
    std::vector<float>& vx, std::vector<float>& vy, std::vector<float>& vz,
    std::vector<float>& mass, std::vector<float>& temp, std::vector<float>& lum,
    std::vector<uint8_t>& type, std::vector<uint8_t>& alive
) {
    auto& d = sys.deviceData();
    CUDA_CHECK(cudaMemcpy(d.pos_x, px.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.pos_y, py.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.pos_z, pz.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vel_x, vx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vel_y, vy.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vel_z, vz.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.mass, mass.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.temperature, temp.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.luminosity, lum.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.type, type.data(), n * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.alive, alive.data(), n * sizeof(uint8_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d.acc_x, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d.acc_y, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d.acc_z, 0, n * sizeof(float)));

    std::vector<float> rad(n, 0.01f);
    std::vector<float> age(n, 0.0f);
    CUDA_CHECK(cudaMemcpy(d.radius, rad.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.age, age.data(), n * sizeof(float), cudaMemcpyHostToDevice));
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

void InitialConditions::bigBang(ParticleSystem& sys, uint32_t n) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::normal_distribution<float> g(0.0f, 1.0f);

    std::vector<float> px(n), py(n), pz(n), vx(n), vy(n), vz(n), mass(n), temp(n), lum(n);
    std::vector<uint8_t> type(n, PARTICLE_STAR), alive(n, 1);

    for (uint32_t i = 0; i < n; i++) {
        float r = 0.5f * std::cbrt(u(rng));
        float th = 6.28318530718f * u(rng);
        float ph = acosf(2.0f * u(rng) - 1.0f);

        px[i] = r * sinf(ph) * cosf(th) + g(rng) * 0.01f;
        py[i] = r * sinf(ph) * sinf(th) + g(rng) * 0.01f;
        pz[i] = r * cosf(ph) + g(rng) * 0.01f;

        vx[i] = 0.5f * px[i];
        vy[i] = 0.5f * py[i];
        vz[i] = 0.5f * pz[i];

        mass[i] = 1.0f;
        temp[i] = 8000.0f + g(rng) * 2000.0f;
        lum[i] = 0.5f + u(rng) * 1.5f;
    }

    uploadToGPU(sys, n, px, py, pz, vx, vy, vz, mass, temp, lum, type, alive);
}

void InitialConditions::galaxyCollision(ParticleSystem& sys, uint32_t n) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::normal_distribution<float> g(0.0f, 1.0f);

    uint32_t half = n / 2;

    std::vector<float> px(n), py(n), pz(n), vx(n), vy(n), vz(n), mass(n), temp(n), lum(n);
    std::vector<uint8_t> type(n, PARTICLE_STAR), alive(n, 1);

    auto makeGalaxy = [&](uint32_t start, uint32_t count, float cx, float cy, float cz,
                          float cvx, float cvy, float cvz,
                          float diskRadius, float bulgeRadius, float tilt) {
        uint32_t bulge = count / 5;
        uint32_t disk = count - bulge;

        for (uint32_t i = 0; i < bulge; i++) {
            uint32_t idx = start + i;
            float r = bulgeRadius * std::cbrt(u(rng));
            float th = 6.28318530718f * u(rng);
            float ph = acosf(2.0f * u(rng) - 1.0f);

            float x = r * sinf(ph) * cosf(th);
            float y = r * sinf(ph) * sinf(th);
            float z = r * cosf(ph);

            px[idx] = cx + x;
            py[idx] = cy + y;
            pz[idx] = cz + z;

            vx[idx] = cvx + g(rng) * 0.15f;
            vy[idx] = cvy + g(rng) * 0.15f;
            vz[idx] = cvz + g(rng) * 0.15f;

            mass[idx] = 1.5f + u(rng) * 2.0f;
            temp[idx] = 4500.0f + u(rng) * 6000.0f;
            lum[idx] = 0.7f + u(rng) * 1.2f;
        }

        for (uint32_t i = 0; i < disk; i++) {
            uint32_t idx = start + bulge + i;
            float r = diskRadius * sqrtf(u(rng));
            float th = 6.28318530718f * u(rng);

            float localX = r * cosf(th);
            float localY = r * sinf(th);
            float localZ = g(rng) * 0.12f;

            float c = cosf(tilt);
            float s = sinf(tilt);

            float rx = localX;
            float ry = localY * c - localZ * s;
            float rz = localY * s + localZ * c;

            px[idx] = cx + rx;
            py[idx] = cy + ry;
            pz[idx] = cz + rz;

            float orb = sqrtf((float)bulge * 1.3f / (r + 0.5f));
            float vxLocal = -orb * sinf(th);
            float vyLocal =  orb * cosf(th);
            float vzLocal = 0.0f;

            float rvx = vxLocal;
            float rvy = vyLocal * c - vzLocal * s;
            float rvz = vyLocal * s + vzLocal * c;

            vx[idx] = cvx + rvx;
            vy[idx] = cvy + rvy;
            vz[idx] = cvz + rvz;

            mass[idx] = 0.5f + u(rng) * 0.7f;
            temp[idx] = 3000.0f + u(rng) * 18000.0f;
            lum[idx] = 0.3f + u(rng) * 1.5f;

            if (u(rng) < 0.08f) type[idx] = PARTICLE_GAS;
        }
    };

    // Much wider initial separation for clearer startup view
    makeGalaxy(0, half, -45.0f, 0.0f, 0.0f, 0.55f, 0.08f, 0.0f, 14.0f, 3.5f, 0.15f);
    makeGalaxy(half, n - half, 45.0f, 0.0f, 0.0f, -0.55f, -0.08f, 0.0f, 14.0f, 3.5f, -0.22f);

    uploadToGPU(sys, n, px, py, pz, vx, vy, vz, mass, temp, lum, type, alive);
}

void InitialConditions::protoCloud(ParticleSystem& sys, uint32_t n) {
    std::mt19937 rng(777);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::normal_distribution<float> g(0.0f, 1.0f);

    std::vector<float> px(n), py(n), pz(n), vx(n), vy(n), vz(n), mass(n), temp(n), lum(n);
    std::vector<uint8_t> type(n, PARTICLE_STAR), alive(n, 1);

    for (uint32_t i = 0; i < n; i++) {
        float r = 30.0f * std::cbrt(u(rng)) * (1.0f + g(rng) * 0.05f);
        float th = 6.28318530718f * u(rng);
        float ph = acosf(2.0f * u(rng) - 1.0f);

        px[i] = r * sinf(ph) * cosf(th);
        py[i] = r * sinf(ph) * sinf(th);
        pz[i] = r * cosf(ph);

        float ang = 0.03f;
        vx[i] = -ang * py[i] + g(rng) * 0.05f;
        vy[i] =  ang * px[i] + g(rng) * 0.05f;
        vz[i] = g(rng) * 0.02f;

        mass[i] = 0.5f + u(rng) * 1.5f;
        temp[i] = 5000.0f + u(rng) * 10000.0f;
        lum[i] = 0.5f + u(rng);

        if (u(rng) < 0.3f) type[i] = PARTICLE_GAS;
    }

    uploadToGPU(sys, n, px, py, pz, vx, vy, vz, mass, temp, lum, type, alive);
}

void InitialConditions::solarSystem(ParticleSystem& sys, uint32_t n) {
    std::mt19937 rng(314);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::normal_distribution<float> g(0.0f, 1.0f);

    std::vector<float> px(n), py(n), pz(n), vx(n), vy(n), vz(n), mass(n), temp(n), lum(n);
    std::vector<uint8_t> type(n, PARTICLE_STAR), alive(n, 1);

    px[0] = py[0] = pz[0] = 0.0f;
    vx[0] = vy[0] = vz[0] = 0.0f;
    mass[0] = 100000.0f;
    temp[0] = 5778.0f;
    lum[0] = 5.0f;
    type[0] = PARTICLE_STAR;

    for (uint32_t i = 1; i < n; i++) {
        float r = 1.0f + u(rng) * 40.0f;
        float th = 6.28318530718f * u(rng);

        px[i] = r * cosf(th);
        py[i] = r * sinf(th);
        pz[i] = g(rng) * 0.1f * r;

        float orb = sqrtf(mass[0] / r);
        vx[i] = -orb * sinf(th);
        vy[i] =  orb * cosf(th);
        vz[i] = 0.0f;

        mass[i] = 0.001f + u(rng) * 0.05f;
        temp[i] = 500.0f + u(rng) * 2000.0f;
        lum[i] = 0.1f + u(rng) * 0.3f;
        type[i] = PARTICLE_GAS;
    }

    uploadToGPU(sys, n, px, py, pz, vx, vy, vz, mass, temp, lum, type, alive);
}

void InitialConditions::randomSphere(ParticleSystem& sys, uint32_t n) {
    sys.initRandom(12345ULL);
}