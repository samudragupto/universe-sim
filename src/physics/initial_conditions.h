#pragma once

#include "particle_system.h"
#include <string>

enum class Scenario { BIG_BANG, GALAXY_COLLISION, PROTOGALACTIC_CLOUD, SOLAR_SYSTEM, RANDOM_SPHERE };

class InitialConditions {
public:
    static void generate(ParticleSystem& sys, Scenario sc, uint32_t count);
private:
    static void bigBang(ParticleSystem& sys, uint32_t n);
    static void galaxyCollision(ParticleSystem& sys, uint32_t n);
    static void protoCloud(ParticleSystem& sys, uint32_t n);
    static void solarSystem(ParticleSystem& sys, uint32_t n);
    static void randomSphere(ParticleSystem& sys, uint32_t n);
};

Scenario scenarioFromString(const std::string& name);