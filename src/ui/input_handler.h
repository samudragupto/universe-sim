#pragma once

#include <GLFW/glfw3.h>
#include "render/camera.h"

class InputHandler {
public:
    InputHandler();

    void init(GLFWwindow* window);
    void processInput(GLFWwindow* window, Camera& camera, float deltaTime);

    bool shouldClose() const { return m_shouldClose; }

    bool isPauseToggled();
    bool isModeToggled();
    bool isScreenshotRequested();
    bool isRecordToggled();
    bool isOverlayToggled();
    bool isTrailsToggled();
    bool isBloomToggled();
    bool isEvolutionToggled();
    bool isVolumetricToggled();
    bool isFullscreenToggled();
    int getScenarioSwitch();

    static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

    static float s_lastX;
    static float s_lastY;
    static float s_xOffset;
    static float s_yOffset;
    static float s_scrollOffset;

    static bool s_firstMouse;
    static bool s_pausePressed;
    static bool s_modePressed;
    static bool s_rightMouseDown;

    static bool s_screenshotPressed;
    static bool s_recordPressed;
    static bool s_overlayPressed;
    static bool s_trailsPressed;
    static bool s_bloomPressed;
    static bool s_evolutionPressed;
    static bool s_volumetricPressed;
    static bool s_fullscreenPressed;

    static int s_scenarioSwitch;

private:
    bool m_shouldClose;
};