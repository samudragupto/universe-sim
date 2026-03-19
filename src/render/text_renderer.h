#pragma once

#include <glad/glad.h>
#include <string>
#include <vector>

struct TextEntry {
    std::string text;
    float x, y;
    float r, g, b, a;
    float scale;
};

class TextRenderer {
public:
    TextRenderer();
    ~TextRenderer();

    void init(int screenWidth, int screenHeight);
    void resize(int screenWidth, int screenHeight);
    void begin();
    void drawText(const std::string& text, float x, float y,
                  float scale = 1.0f, float r = 1.0f, float g = 1.0f, float b = 1.0f, float a = 1.0f);
    void end();
    void cleanup();

private:
    void createFontTexture();
    void buildVertices();

    GLuint m_vao;
    GLuint m_vbo;
    GLuint m_program;
    GLuint m_fontTexture;

    int m_screenWidth;
    int m_screenHeight;

    std::vector<TextEntry> m_entries;
    std::vector<float> m_vertices;
};