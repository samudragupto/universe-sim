#pragma once

#include <glad/glad.h>
#include <string>
#include <cstdint>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>

class ScreenCapture {
public:
    ScreenCapture();
    ~ScreenCapture();

    void init(int width, int height);
    void resize(int width, int height);

    void saveScreenshot(const std::string& filename);
    void beginRecording(const std::string& directory, int targetFPS = 60);
    void captureFrame();
    void endRecording();
    bool isRecording() const { return m_recording; }
    uint32_t getRecordedFrames() const { return m_recordedFrames; }

private:
    void savePixelsToPNG(const std::string& filename, const std::vector<uint8_t>& pixels, int w, int h);

    int m_width;
    int m_height;
    GLuint m_pbo[2];
    int m_pboIndex;

    std::atomic<bool> m_recording;
    std::string m_recordDir;
    uint32_t m_recordedFrames;
    int m_targetFPS;
};