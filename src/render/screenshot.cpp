#include "screenshot.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cstdio>
#include <cstring>
#include <filesystem>

ScreenCapture::ScreenCapture()
    : m_width(0), m_height(0), m_pboIndex(0)
    , m_recording(false), m_recordedFrames(0), m_targetFPS(60) {
    m_pbo[0] = m_pbo[1] = 0;
}

ScreenCapture::~ScreenCapture() {
    if (m_pbo[0]) glDeleteBuffers(2, m_pbo);
}

void ScreenCapture::init(int width, int height) {
    m_width = width;
    m_height = height;

    glGenBuffers(2, m_pbo);
    for (int i = 0; i < 2; i++) {
        glBindBuffer(GL_PIXEL_PACK_BUFFER, m_pbo[i]);
        glBufferData(GL_PIXEL_PACK_BUFFER, width * height * 3, nullptr, GL_STREAM_READ);
    }
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

void ScreenCapture::resize(int width, int height) {
    m_width = width;
    m_height = height;
    if (m_pbo[0]) glDeleteBuffers(2, m_pbo);
    glGenBuffers(2, m_pbo);
    for (int i = 0; i < 2; i++) {
        glBindBuffer(GL_PIXEL_PACK_BUFFER, m_pbo[i]);
        glBufferData(GL_PIXEL_PACK_BUFFER, width * height * 3, nullptr, GL_STREAM_READ);
    }
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

void ScreenCapture::saveScreenshot(const std::string& filename) {
    std::vector<uint8_t> pixels(m_width * m_height * 3);
    glReadPixels(0, 0, m_width, m_height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    savePixelsToPNG(filename, pixels, m_width, m_height);
    printf("Screenshot saved: %s\n", filename.c_str());
}

void ScreenCapture::beginRecording(const std::string& directory, int targetFPS) {
    m_recordDir = directory;
    m_targetFPS = targetFPS;
    m_recordedFrames = 0;
    m_recording = true;

    std::filesystem::create_directories(directory);
    printf("Recording started: %s @ %d FPS\n", directory.c_str(), targetFPS);
}

void ScreenCapture::captureFrame() {
    if (!m_recording) return;

    char filename[512];
    snprintf(filename, sizeof(filename), "%s/frame_%06u.png", m_recordDir.c_str(), m_recordedFrames);

    std::vector<uint8_t> pixels(m_width * m_height * 3);
    glReadPixels(0, 0, m_width, m_height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    savePixelsToPNG(filename, pixels, m_width, m_height);

    m_recordedFrames++;
}

void ScreenCapture::endRecording() {
    m_recording = false;
    printf("Recording stopped: %u frames captured\n", m_recordedFrames);
    printf("Convert with: ffmpeg -framerate 60 -i %s/frame_%%06d.png -c:v libx264 -pix_fmt yuv420p output.mp4\n",
           m_recordDir.c_str());
}

void ScreenCapture::savePixelsToPNG(const std::string& filename,
                                     const std::vector<uint8_t>& pixels,
                                     int w, int h) {
    std::vector<uint8_t> flipped(w * h * 3);
    for (int y = 0; y < h; y++) {
        memcpy(&flipped[y * w * 3], &pixels[(h - 1 - y) * w * 3], w * 3);
    }
    stbi_write_png(filename.c_str(), w, h, 3, flipped.data(), w * 3);
}