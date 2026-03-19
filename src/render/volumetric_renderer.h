#pragma once

#include <glad/glad.h>
#include "shader_manager.h"
#include "camera.h"
#include "cuda/density_field.cuh"

class VolumetricRenderer {
public:
    VolumetricRenderer();
    ~VolumetricRenderer();

    void init(int fieldRes, ShaderManager& shaders);
    void cleanup();
    void updateTexture(const DensityField& df, cudaStream_t stream);
    void render(ShaderManager& shaders, const Camera& camera, const DensityField& df,
                GLuint targetFBO, int width, int height);

    bool isEnabled() const { return m_enabled; }
    void setEnabled(bool e) { m_enabled = e; }
    GLuint getResultTexture() const { return m_resultTexture; }

private:
    GLuint m_densityTexture3D;
    GLuint m_resultFBO;
    GLuint m_resultTexture;
    GLuint m_quadVAO;
    int m_fieldRes;
    bool m_enabled;
    bool m_initialized;
};