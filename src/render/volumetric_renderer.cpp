#include <glad/glad.h>
#include "volumetric_renderer.h"
#include "cuda/cuda_utils.h"
#include <cstdio>
#include <vector>

VolumetricRenderer::VolumetricRenderer()
    : m_densityTexture3D(0), m_resultFBO(0), m_resultTexture(0)
    , m_quadVAO(0), m_fieldRes(0), m_enabled(false), m_initialized(false) {}

VolumetricRenderer::~VolumetricRenderer() { cleanup(); }

void VolumetricRenderer::init(int fieldRes, ShaderManager& shaders) {
    m_fieldRes = fieldRes;
    m_enabled = true;
    m_initialized = true;

    // We assume shaders are already loaded via shader manager
    // "volumetric" -> shaders/volumetric.vert, shaders/volumetric.frag
    shaders.loadProgram("volumetric", "shaders/volumetric.vert", "shaders/volumetric.frag");

    glGenTextures(1, &m_densityTexture3D);
    glBindTexture(GL_TEXTURE_3D, m_densityTexture3D);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, fieldRes, fieldRes, fieldRes, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);

    glGenVertexArrays(1, &m_quadVAO);
}

void VolumetricRenderer::cleanup() {
    if (m_densityTexture3D) glDeleteTextures(1, &m_densityTexture3D);
    if (m_resultFBO) glDeleteFramebuffers(1, &m_resultFBO);
    if (m_resultTexture) glDeleteTextures(1, &m_resultTexture);
    if (m_quadVAO) glDeleteVertexArrays(1, &m_quadVAO);
    m_densityTexture3D = 0; m_resultFBO = 0; m_resultTexture = 0; m_quadVAO = 0;
    m_initialized = false;
}

void VolumetricRenderer::updateTexture(const DensityField& df, cudaStream_t stream) {
    if (!m_initialized || !m_enabled || !df.allocated) return;

    int total = df.resX * df.resY * df.resZ;
    std::vector<float> hostData(total);
    
    // Copy data from GPU back to Host for OpenGL 3D texture upload
    CUDA_CHECK(cudaMemcpyAsync(hostData.data(), df.data, total * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    glBindTexture(GL_TEXTURE_3D, m_densityTexture3D);
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, df.resX, df.resY, df.resZ,
                    GL_RED, GL_FLOAT, hostData.data());
}

void VolumetricRenderer::render(ShaderManager& shaders, const Camera& camera,
                                 const DensityField& df,
                                 GLuint targetFBO, int width, int height) {
    if (!m_initialized || !m_enabled || !df.allocated) return;

    glBindFramebuffer(GL_FRAMEBUFFER, targetFBO);
    glViewport(0, 0, width, height);

    GLuint prog = shaders.getProgram("volumetric");
    if (!prog) return;
    glUseProgram(prog);

    glm::mat4 view = camera.getViewMatrix();
    glm::mat4 proj = camera.getProjectionMatrix();
    glm::mat4 invView = glm::inverse(view);
    glm::mat4 invProj = glm::inverse(proj);
    glm::vec3 camPos = camera.getPosition();

    glUniformMatrix4fv(glGetUniformLocation(prog, "uInvView"), 1, GL_FALSE, &invView[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(prog, "uInvProjection"), 1, GL_FALSE, &invProj[0][0]);
    glUniform3f(glGetUniformLocation(prog, "uCameraPos"), camPos.x, camPos.y, camPos.z);
    glUniform3f(glGetUniformLocation(prog, "uFieldMin"), df.minX, df.minY, df.minZ);
    glUniform3f(glGetUniformLocation(prog, "uFieldMax"), df.maxX, df.maxY, df.maxZ);
    glUniform1f(glGetUniformLocation(prog, "uDensityScale"), 5.0f);
    glUniform1f(glGetUniformLocation(prog, "uStepSize"), 0.5f);
    glUniform1i(glGetUniformLocation(prog, "uMaxSteps"), 128);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, m_densityTexture3D);
    glUniform1i(glGetUniformLocation(prog, "uDensityField"), 0);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    glBindVertexArray(m_quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);

    glDepthMask(GL_TRUE);
}