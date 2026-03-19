#pragma once

#include <GLFW/glfw3.h>
#include "render/camera.h"

class InputHandler {
public:
    InputHandler();
    void init(GLFWwindow* w);
    void processInput(GLFWwindow* w, Camera& cam, float dt);
    bool shouldClose() const { return m_close; }
    bool isPauseToggled();
    bool isModeToggled();
    bool isScreenshotRequested();
    bool isRecordToggled();
    bool isOverlayToggled();
    bool isTrailsToggled();
    bool isBloomToggled();
    bool isEvolutionToggled();
    bool isVolumetricToggled();
    int getScenarioSwitch();

    static void mouseCallback(GLFWwindow* w, double x, double y);
    static void scrollCallback(GLFWwindow* w, double x, double y);
    static void keyCallback(GLFWwindow* w, int key, int sc, int act, int mods);

    static float s_lx, s_ly, s_xo, s_yo, s_so;
    static bool s_first, s_pause, s_mode, s_rmb;
    static bool s_shot, s_rec, s_overlay, s_trails, s_bloom, s_evol, s_vol;
    static int s_scen;
private:
    bool m_close;
};