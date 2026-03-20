#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>
#include "render/renderer.h"
#include "render/camera.h"
#include "physics/simulation.h"
#include "physics/particle_system.h"
#include "physics/initial_conditions.h"
#include "ui/input_handler.h"

struct AppConfig {
    uint32_t particleCount;
    float G, softening, timestep, theta, mergeDistance;
    bool bruteForce, evolution, adaptiveTimestep, adaptiveTheta;
    bool volumetricEnabled;
    int densityFieldRes;
    int winW, winH;
    std::string scenario;
    bool bloom; float bloomThresh, bloomIntensity, exposure;
    bool trails; uint32_t trailLen, maxTrailP;
    float vignette, chromatic;
    float camSpeed, camSens, camFOV, nearP, farP;
};

class Application {
public:
    Application();
    ~Application();

    bool init();
    void run();
    void shutdown();

private:
    bool initWindow();
    bool initCUDA();
    void loadConfig();
    void mainLoop();
    void resetSimulation(Scenario sc);
    void toggleFullscreen();

    static void fbCallback(GLFWwindow* w, int width, int height);

    GLFWwindow* m_window;
    AppConfig m_cfg;
    Renderer m_renderer;
    Camera m_camera;
    Simulation m_sim;
    ParticleSystem m_particles;
    InputHandler m_input;

    float m_lastTime, m_dt;
    uint64_t m_frameCount;
    float m_fpsTimer, m_fps;
    bool m_initialized;

    bool m_fullscreen;
    int m_windowedX;
    int m_windowedY;
    int m_windowedW;
    int m_windowedH;
};