#include "renderer.h"
#include "cuda/cuda_utils.h"
#include "cuda/kernels.cuh"
#include <cstdio>
#include <glm/gtc/matrix_transform.hpp>

Renderer::Renderer()
    : m_particleVAO(0), m_particleVBO(0), m_maxParticles(0), m_renderedCount(0)
    , m_renderStream(0), m_trailVAO(0), m_trailVBO(0), m_trailsAllocated(false)
    , m_overlayVisible(true) {}

Renderer::~Renderer() { shutdown(); }

bool Renderer::init(const RenderConfig& cfg) {
    m_config = cfg;
    m_shaders.loadProgram("particle", "shaders/particle.vert", "shaders/particle.frag");
    m_shaders.loadProgram("trail", "shaders/trail.vert", "shaders/trail.frag");
    m_postProcess.init(cfg.width, cfg.height, m_shaders);
    m_textRenderer.init(cfg.width, cfg.height);
    m_screenCapture.init(cfg.width, cfg.height);
    if (cfg.volumetricEnabled) m_volumetric.init(cfg.densityFieldRes, m_shaders);
    CUDA_CHECK(cudaStreamCreate(&m_renderStream));
    glEnable(GL_PROGRAM_POINT_SIZE);
    return true;
}

void Renderer::shutdown() {
    m_interop.unregisterBuffer();
    if (m_particleVAO) glDeleteVertexArrays(1, &m_particleVAO);
    if (m_particleVBO) glDeleteBuffers(1, &m_particleVBO);
    if (m_trailVAO) glDeleteVertexArrays(1, &m_trailVAO);
    if (m_trailVBO) glDeleteBuffers(1, &m_trailVBO);
    if (m_trailsAllocated) trailDeallocate(m_trailData);
    m_volumetric.cleanup();
    m_shaders.deleteAll();
    m_postProcess.cleanup();
    m_textRenderer.cleanup();
    if (m_renderStream) { cudaStreamDestroy(m_renderStream); m_renderStream = 0; }
    m_particleVAO = m_particleVBO = m_trailVAO = m_trailVBO = 0;
    m_trailsAllocated = false;
}

void Renderer::createParticleBuffers(uint32_t n) {
    m_maxParticles = n;
    glGenVertexArrays(1, &m_particleVAO);
    glGenBuffers(1, &m_particleVBO);
    glBindVertexArray(m_particleVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_particleVBO);
    glBufferData(GL_ARRAY_BUFFER, n * sizeof(RenderVertex), nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void*)(7*sizeof(float)));
    glBindVertexArray(0);
}

void Renderer::createTrailBuffers() {
    uint32_t tv = m_trailData.totalVertices;
    glGenVertexArrays(1, &m_trailVAO);
    glGenBuffers(1, &m_trailVBO);
    glBindVertexArray(m_trailVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_trailVBO);
    glBufferData(GL_ARRAY_BUFFER, tv * sizeof(TrailVertex), nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(TrailVertex), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(TrailVertex), (void*)(3*sizeof(float)));
    glBindVertexArray(0);
}

void Renderer::setupInterop(ParticleSystem& ps) {
    createParticleBuffers(ps.count());
    m_interop.registerBuffer(m_particleVBO, ps.count());
    if (m_config.trailsEnabled) {
        uint32_t tp = m_config.maxTrailParticles > 0 ? m_config.maxTrailParticles : ps.count();
        if (tp > ps.count()) tp = ps.count();
        trailAllocate(m_trailData, tp, m_config.trailLength);
        m_trailsAllocated = true;
        createTrailBuffers();
    }
}

void Renderer::updateRenderBuffer(ParticleSystem& ps) {
    RenderVertex* ptr = m_interop.mapBuffer(m_renderStream);
    if (!ptr) return;
    launchComputeColors(ps.deviceData(), ptr, ps.count(), m_renderStream);
    CUDA_CHECK(cudaStreamSynchronize(m_renderStream));
    m_interop.unmapBuffer(m_renderStream);
    m_renderedCount = ps.count();
}

void Renderer::updateTrails(ParticleSystem& ps) {
    if (!m_config.trailsEnabled || !m_trailsAllocated) return;
    launchUpdateTrails(ps.deviceData(), m_trailData, ps.count(), m_renderStream);
    CUDA_CHECK(cudaStreamSynchronize(m_renderStream));
    uint32_t tv = m_trailData.totalVertices;
    std::vector<TrailVertex> host(tv);
    CUDA_CHECK(cudaMemcpy(host.data(), m_trailData.vertices, tv*sizeof(TrailVertex), cudaMemcpyDeviceToHost));
    glBindBuffer(GL_ARRAY_BUFFER, m_trailVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, tv*sizeof(TrailVertex), host.data());
}

void Renderer::updateVolumetric(const DensityField& df) {
    if (m_config.volumetricEnabled) m_volumetric.updateTexture(df, m_renderStream);
}

void Renderer::findBlackHoles(ParticleSystem& ps, const Camera& cam) {
    m_blackHoles.clear();
    auto& d = ps.deviceData();
    uint32_t n = ps.count(), sc = (n > 5000) ? 5000 : n;
    std::vector<float> hpx(sc), hpy(sc), hpz(sc), hm(sc);
    std::vector<uint8_t> ht(sc), ha(sc);
    CUDA_CHECK(cudaMemcpy(hpx.data(), d.pos_x, sc*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hpy.data(), d.pos_y, sc*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hpz.data(), d.pos_z, sc*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hm.data(), d.mass, sc*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ht.data(), d.type, sc, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ha.data(), d.alive, sc, cudaMemcpyDeviceToHost));
    glm::mat4 vp = cam.getProjectionMatrix() * cam.getViewMatrix();
    for (uint32_t i = 0; i < sc && m_blackHoles.size() < 16; i++) {
        if (ht[i] != 2 || !ha[i]) continue;
        glm::vec4 cl = vp * glm::vec4(hpx[i], hpy[i], hpz[i], 1.0f);
        if (cl.w <= 0) continue;
        glm::vec3 ndc = glm::vec3(cl)/cl.w;
        if (ndc.x<-1||ndc.x>1||ndc.y<-1||ndc.y>1) continue;
        BlackHoleScreenData bh;
        bh.screenPos = glm::vec2(ndc.x*.5f+.5f, ndc.y*.5f+.5f);
        bh.mass = hm[i]; bh.screenRadius = hm[i]*.001f/cl.w*(float)m_config.height;
        m_blackHoles.push_back(bh);
    }
}

void Renderer::beginFrame() { m_postProcess.bindHDRFramebuffer(); }

void Renderer::renderParticles(const Camera& cam, uint32_t count) {
    GLuint prog = m_shaders.getProgram("particle");
    glUseProgram(prog);
    glm::mat4 v = cam.getViewMatrix(), p = cam.getProjectionMatrix();
    glUniformMatrix4fv(glGetUniformLocation(prog, "uView"), 1, GL_FALSE, &v[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(prog, "uProjection"), 1, GL_FALSE, &p[0][0]);
    glUniform1f(glGetUniformLocation(prog, "uScreenHeight"), (float)m_config.height);
    glUniform3fv(glGetUniformLocation(prog, "uCameraPos"), 1, &cam.getPosition()[0]);
    glUniform1f(glGetUniformLocation(prog, "uMaxPointSize"), 64.0f);
    glUniform1f(glGetUniformLocation(prog, "uMinPointSize"), 1.0f);
    glEnable(GL_BLEND); glBlendFunc(GL_ONE, GL_ONE); glDepthMask(GL_FALSE);
    glBindVertexArray(m_particleVAO);
    glDrawArrays(GL_POINTS, 0, count);
    glBindVertexArray(0);
    glDepthMask(GL_TRUE);
}

void Renderer::renderTrails(const Camera& cam) {
    if (!m_config.trailsEnabled || !m_trailsAllocated) return;
    GLuint prog = m_shaders.getProgram("trail");
    glUseProgram(prog);
    glm::mat4 v = cam.getViewMatrix(), p = cam.getProjectionMatrix();
    glUniformMatrix4fv(glGetUniformLocation(prog, "uView"), 1, GL_FALSE, &v[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(prog, "uProjection"), 1, GL_FALSE, &p[0][0]);
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE); glDepthMask(GL_FALSE);
    glPointSize(1.0f);
    glBindVertexArray(m_trailVAO);
    glDrawArrays(GL_POINTS, 0, m_trailData.totalVertices);
    glBindVertexArray(0);
    glDepthMask(GL_TRUE);
}

void Renderer::renderVolumetric(const Camera& cam, const DensityField& df) {
    if (!m_config.volumetricEnabled) return;
    m_volumetric.render(m_shaders, cam, df, m_postProcess.getHDRFBO(), m_config.width, m_config.height);
}

void Renderer::renderOverlay(const OverlayStats& st) {
    if (!m_overlayVisible) return;
    m_textRenderer.begin();
    char b[256]; float y = 10, s = 2, dy = 18;

    snprintf(b, 256, "FPS: %.1f  Frame: %.1fms", st.fps, st.frameTimeMs);
    m_textRenderer.drawText(b, 10, y, s, 0, 1, 0); y += dy;
    snprintf(b, 256, "Particles: %u  Alive: %u", st.particleCount, st.diag.aliveCount);
    m_textRenderer.drawText(b, 10, y, s, 1, 1, 1); y += dy;
    snprintf(b, 256, "Stars:%u Gas:%u BH:%u DM:%u NS:%u",
             st.diag.starCount, st.diag.gasCount, st.diag.bhCount, st.diag.dmCount, st.diag.nsCount);
    m_textRenderer.drawText(b, 10, y, s, .8f, .8f, 1); y += dy;
    snprintf(b, 256, "Time: %.3f  Step: %lu", st.simTime, (unsigned long)st.simStep);
    m_textRenderer.drawText(b, 10, y, s, .8f, .8f, 1); y += dy;
    snprintf(b, 256, "Tree:%.1fms Force:%.1fms Int:%.1fms Tot:%.1fms",
             st.treeBuildMs, st.forceCalcMs, st.integrationMs, st.totalStepMs);
    m_textRenderer.drawText(b, 10, y, s, 1, .8f, .3f); y += dy;
    snprintf(b, 256, "KE: %.1f  MaxV: %.2f  MaxA: %.2f",
             st.diag.totalKineticEnergy, st.diag.maxVelocity, st.diag.maxAcceleration);
    m_textRenderer.drawText(b, 10, y, s, .5f, 1, .5f); y += dy;
    snprintf(b, 256, "Camera: %s  %s", st.cameraMode.c_str(), st.paused ? "[PAUSED]" : "[RUNNING]");
    m_textRenderer.drawText(b, 10, y, s, .5f, 1, .5f); y += dy;
    snprintf(b, 256, "Bloom:%s Trails:%s Evol:%s Vol:%s",
             st.bloomOn?"ON":"OFF", st.trailsOn?"ON":"OFF", st.evolutionOn?"ON":"OFF", st.volumetricOn?"ON":"OFF");
    m_textRenderer.drawText(b, 10, y, s, .7f, .7f, .7f); y += dy;
    if (st.recording) {
        snprintf(b, 256, "REC: %u frames", st.recordedFrames);
        m_textRenderer.drawText(b, 10, y, s, 1, .2f, .2f);
    }
    y = (float)m_config.height - 30;
    m_textRenderer.drawText("SPACE:Pause TAB:Camera F2:Shot F3:Rec F4:Overlay B:Bloom T:Trails V:Evol G:Vol", 10, y, 1.5f, .5f, .5f, .5f);

    m_textRenderer.end();
}

void Renderer::endFrame(GLFWwindow* w, const Camera& cam) {
    m_postProcess.render(m_shaders, m_config.bloomThreshold, m_config.bloomIntensity,
                          m_config.bloomEnabled, m_config.exposure,
                          m_blackHoles, m_config.vignetteStrength, m_config.chromaticStrength);
    captureFrameIfRecording();
}

void Renderer::resize(int w, int h) {
    m_config.width = w; m_config.height = h;
    m_postProcess.resize(w, h);
    m_textRenderer.resize(w, h);
    m_screenCapture.resize(w, h);
}

void Renderer::takeScreenshot(const std::string& fn) { m_screenCapture.saveScreenshot(fn); }
void Renderer::toggleRecording(const std::string& d) {
    if (m_screenCapture.isRecording()) m_screenCapture.endRecording();
    else m_screenCapture.beginRecording(d);
}
void Renderer::captureFrameIfRecording() { if (m_screenCapture.isRecording()) m_screenCapture.captureFrame(); }
bool Renderer::isRecording() const { return m_screenCapture.isRecording(); }
uint32_t Renderer::getRecordedFrames() const { return m_screenCapture.getRecordedFrames(); }