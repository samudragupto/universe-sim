#pragma once

#include <glad/glad.h>
#include "shader_manager.h"
#include <glm/glm.hpp>
#include <vector>

struct BlackHoleScreenData {
    glm::vec2 screenPos;
    float mass;
    float screenRadius;
};

class PostProcess {
public:
    PostProcess();
    ~PostProcess();

    void init(int w, int h, ShaderManager& shaders);
    void resize(int w, int h);
    void bindHDRFramebuffer();
    void render(ShaderManager& shaders, float bloomThresh, float bloomInt,
                bool bloomOn, float exposure,
                const std::vector<BlackHoleScreenData>& bhs,
                float vignette, float chromatic);
    void cleanup();

    GLuint getHDRTexture() const { return m_hdrColorTexture; }
    GLuint getHDRFBO() const { return m_hdrFBO; } // <--- This fixes the "no member getHDRFBO" error

private:
    void createFramebuffers(int w, int h);
    void renderFullscreenQuad();

    int m_width;
    int m_height;

    GLuint m_hdrFBO;
    GLuint m_hdrColorTexture;
    GLuint m_hdrDepthRBO;

    GLuint m_bloomExtractFBO;
    GLuint m_bloomExtractTexture;

    GLuint m_pingpongFBO[2];
    GLuint m_pingpongTexture[2];

    GLuint m_lensingFBO;
    GLuint m_lensingTexture;

    GLuint m_quadVAO;
};