#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "shader_manager.h"
#include "post_process.h"
#include "camera.h"
#include "text_renderer.h"
#include "screenshot.h"
#include "volumetric_renderer.h"
#include "cuda/interop.cuh"
#include "cuda/trails.cuh"
#include "cuda/diagnostics.cuh"
#include "cuda/density_field.cuh"
#include "physics/particle_system.h"
#include <vector>
#include <string>

struct RenderConfig {
    int width, height;
    bool bloomEnabled;
    float bloomThreshold, bloomIntensity, exposure;
    bool trailsEnabled;
    uint32_t trailLength, maxTrailParticles;
    float vignetteStrength, chromaticStrength;
    bool volumetricEnabled;
    int densityFieldRes;
};

struct OverlayStats {
    float fps;
    uint32_t particleCount;
    float simTime;
    uint64_t simStep;
    float treeBuildMs, forceCalcMs, integrationMs, totalStepMs, frameTimeMs;
    bool paused, recording;
    uint32_t recordedFrames;
    std::string cameraMode, scenario;
    bool bloomOn, trailsOn, evolutionOn, volumetricOn;
    SimDiagnostics diag;
};

class Renderer {
public:
    Renderer();
    ~Renderer();
    bool init(const RenderConfig& config);
    void shutdown();
    void beginFrame();
    void renderParticles(const Camera& camera, uint32_t count);
    void renderTrails(const Camera& camera);
    void renderVolumetric(const Camera& camera, const DensityField& df);
    void renderOverlay(const OverlayStats& stats);
    void endFrame(GLFWwindow* window, const Camera& camera);
    void resize(int width, int height);
    void setupInterop(ParticleSystem& ps);
    void updateRenderBuffer(ParticleSystem& ps);
    void updateTrails(ParticleSystem& ps);
    void updateVolumetric(const DensityField& df);
    void findBlackHoles(ParticleSystem& ps, const Camera& camera);
    void takeScreenshot(const std::string& fn);
    void toggleRecording(const std::string& dir);
    void captureFrameIfRecording();
    bool isRecording() const;
    uint32_t getRecordedFrames() const;
    ShaderManager& shaderManager() { return m_shaders; }
    RenderConfig& config() { return m_config; }
    bool isOverlayVisible() const { return m_overlayVisible; }
    void setOverlayVisible(bool v) { m_overlayVisible = v; }

private:
    void createParticleBuffers(uint32_t maxCount);
    void createTrailBuffers();
    RenderConfig m_config;
    ShaderManager m_shaders;
    PostProcess m_postProcess;
    CUDAGLInterop m_interop;
    TextRenderer m_textRenderer;
    ScreenCapture m_screenCapture;
    VolumetricRenderer m_volumetric;
    GLuint m_particleVAO, m_particleVBO;
    uint32_t m_maxParticles, m_renderedCount;
    cudaStream_t m_renderStream;
    TrailData m_trailData;
    GLuint m_trailVAO, m_trailVBO;
    bool m_trailsAllocated;
    std::vector<BlackHoleScreenData> m_blackHoles;
    bool m_overlayVisible;
};