#include "post_process.h"
#include <cstdio>

PostProcess::PostProcess()
    : m_width(0), m_height(0), m_hdrFBO(0), m_hdrColorTexture(0), m_hdrDepthRBO(0)
    , m_bloomExtractFBO(0), m_bloomExtractTexture(0), m_lensingFBO(0), m_lensingTexture(0), m_quadVAO(0) {
    m_pingpongFBO[0] = m_pingpongFBO[1] = 0;
    m_pingpongTexture[0] = m_pingpongTexture[1] = 0;
}

PostProcess::~PostProcess() {
    cleanup();
}

void PostProcess::init(int w, int h, ShaderManager& shaders) {
    m_width = w;
    m_height = h;

    glGenVertexArrays(1, &m_quadVAO);

    shaders.loadProgram("bloom_extract", "shaders/fullscreen.vert", "shaders/bloom_extract.frag");
    shaders.loadProgram("blur", "shaders/fullscreen.vert", "shaders/blur.frag");
    shaders.loadProgram("composite", "shaders/fullscreen.vert", "shaders/composite.frag");
    shaders.loadProgram("tonemap", "shaders/fullscreen.vert", "shaders/tonemap.frag");
    shaders.loadProgram("lensing", "shaders/fullscreen.vert", "shaders/lensing.frag");
    shaders.loadProgram("vignette", "shaders/fullscreen.vert", "shaders/vignette.frag");

    createFramebuffers(w, h);
}

void PostProcess::resize(int w, int h) {
    cleanup();
    m_width = w;
    m_height = h;
    glGenVertexArrays(1, &m_quadVAO);
    createFramebuffers(w, h);
}

void PostProcess::createFramebuffers(int w, int h) {
    auto createFBOTex = [](GLuint& fbo, GLuint& tex, int width, int height, bool withDepth = false, GLuint* rbo = nullptr) {
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        if (withDepth && rbo) {
            glGenRenderbuffers(1, rbo);
            glBindRenderbuffer(GL_RENDERBUFFER, *rbo);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, *rbo);
        }
    };

    createFBOTex(m_hdrFBO, m_hdrColorTexture, w, h, true, &m_hdrDepthRBO);
    createFBOTex(m_bloomExtractFBO, m_bloomExtractTexture, w / 2, h / 2);
    for (int i = 0; i < 2; i++) {
        createFBOTex(m_pingpongFBO[i], m_pingpongTexture[i], w / 2, h / 2);
    }
    createFBOTex(m_lensingFBO, m_lensingTexture, w, h);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void PostProcess::bindHDRFramebuffer() {
    glBindFramebuffer(GL_FRAMEBUFFER, m_hdrFBO);
    glViewport(0, 0, m_width, m_height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void PostProcess::render(ShaderManager& shaders, float bloomThresh, float bloomInt,
                         bool bloomOn, float exposure,
                         const std::vector<BlackHoleScreenData>& bhs,
                         float vignette, float chromatic) {
    GLuint currentScene = m_hdrColorTexture;

    if (!bhs.empty()) {
        glBindFramebuffer(GL_FRAMEBUFFER, m_lensingFBO);
        glViewport(0, 0, m_width, m_height);
        glClear(GL_COLOR_BUFFER_BIT);

        GLuint lensProg = shaders.getProgram("lensing");
        glUseProgram(lensProg);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, currentScene);
        glUniform1i(glGetUniformLocation(lensProg, "uScene"), 0);

        int bhCount = (int)bhs.size();
        if (bhCount > 16) bhCount = 16;
        glUniform1i(glGetUniformLocation(lensProg, "uBHCount"), bhCount);
        glUniform2f(glGetUniformLocation(lensProg, "uResolution"), (float)m_width, (float)m_height);

        for (int i = 0; i < bhCount; i++) {
            char buf[64];
            snprintf(buf, sizeof(buf), "uBHScreenPos[%d]", i);
            glUniform2f(glGetUniformLocation(lensProg, buf), bhs[i].screenPos.x, bhs[i].screenPos.y);
            snprintf(buf, sizeof(buf), "uBHMass[%d]", i);
            glUniform1f(glGetUniformLocation(lensProg, buf), bhs[i].mass);
            snprintf(buf, sizeof(buf), "uBHScreenRadius[%d]", i);
            glUniform1f(glGetUniformLocation(lensProg, buf), bhs[i].screenRadius);
        }
        renderFullscreenQuad();
        currentScene = m_lensingTexture;
    }

    GLuint bloomTex = m_pingpongTexture[0];

    if (bloomOn) {
        glBindFramebuffer(GL_FRAMEBUFFER, m_bloomExtractFBO);
        glViewport(0, 0, m_width / 2, m_height / 2);
        glClear(GL_COLOR_BUFFER_BIT);
        GLuint extractProg = shaders.getProgram("bloom_extract");
        glUseProgram(extractProg);
        glUniform1f(glGetUniformLocation(extractProg, "uThreshold"), bloomThresh);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, currentScene);
        glUniform1i(glGetUniformLocation(extractProg, "uHDRBuffer"), 0);
        renderFullscreenQuad();

        bool horizontal = true;
        int blurPasses = 10;
        GLuint blurProg = shaders.getProgram("blur");
        glUseProgram(blurProg);
        bool firstIteration = true;
        for (int i = 0; i < blurPasses; i++) {
            glBindFramebuffer(GL_FRAMEBUFFER, m_pingpongFBO[horizontal ? 1 : 0]);
            glViewport(0, 0, m_width / 2, m_height / 2);
            glUniform1i(glGetUniformLocation(blurProg, "uHorizontal"), horizontal ? 1 : 0);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, firstIteration ? m_bloomExtractTexture : m_pingpongTexture[horizontal ? 0 : 1]);
            glUniform1i(glGetUniformLocation(blurProg, "uImage"), 0);
            renderFullscreenQuad();
            horizontal = !horizontal;
            firstIteration = false;
        }
        bloomTex = m_pingpongTexture[!horizontal ? 1 : 0];
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, m_width, m_height);
    glClear(GL_COLOR_BUFFER_BIT);

    if (bloomOn) {
        GLuint compositeProg = shaders.getProgram("composite");
        glUseProgram(compositeProg);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, currentScene);
        glUniform1i(glGetUniformLocation(compositeProg, "uScene"), 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, bloomTex);
        glUniform1i(glGetUniformLocation(compositeProg, "uBloom"), 1);
        glUniform1f(glGetUniformLocation(compositeProg, "uBloomIntensity"), bloomInt);
        glUniform1f(glGetUniformLocation(compositeProg, "uExposure"), exposure);
        renderFullscreenQuad();
    } else {
        GLuint tonemapProg = shaders.getProgram("tonemap");
        glUseProgram(tonemapProg);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, currentScene);
        glUniform1i(glGetUniformLocation(tonemapProg, "uHDRBuffer"), 0);
        renderFullscreenQuad();
    }

    if (vignette > 0.0f || chromatic > 0.0f) {
        GLuint vignetteProg = shaders.getProgram("vignette");
        if (vignetteProg) {
            glUseProgram(vignetteProg);
            glUniform1f(glGetUniformLocation(vignetteProg, "uVignetteStrength"), vignette);
            glUniform1f(glGetUniformLocation(vignetteProg, "uChromaticStrength"), chromatic);
        }
    }
}

void PostProcess::renderFullscreenQuad() {
    glBindVertexArray(m_quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
}

void PostProcess::cleanup() {
    if (m_hdrFBO) glDeleteFramebuffers(1, &m_hdrFBO);
    if (m_hdrColorTexture) glDeleteTextures(1, &m_hdrColorTexture);
    if (m_hdrDepthRBO) glDeleteRenderbuffers(1, &m_hdrDepthRBO);
    if (m_bloomExtractFBO) glDeleteFramebuffers(1, &m_bloomExtractFBO);
    if (m_bloomExtractTexture) glDeleteTextures(1, &m_bloomExtractTexture);
    
    if (m_pingpongFBO[0]) glDeleteFramebuffers(2, m_pingpongFBO);
    if (m_pingpongTexture[0]) glDeleteTextures(2, m_pingpongTexture);
    
    if (m_lensingFBO) glDeleteFramebuffers(1, &m_lensingFBO);
    if (m_lensingTexture) glDeleteTextures(1, &m_lensingTexture);
    if (m_quadVAO) glDeleteVertexArrays(1, &m_quadVAO);

    m_hdrFBO = 0; m_hdrColorTexture = 0; m_hdrDepthRBO = 0;
    m_bloomExtractFBO = 0; m_bloomExtractTexture = 0;
    m_pingpongFBO[0] = 0; m_pingpongFBO[1] = 0;
    m_pingpongTexture[0] = 0; m_pingpongTexture[1] = 0;
    m_lensingFBO = 0; m_lensingTexture = 0;
    m_quadVAO = 0;
}