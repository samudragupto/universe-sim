#include "input_handler.h"

float InputHandler::s_lastX = 960.0f;
float InputHandler::s_lastY = 540.0f;
float InputHandler::s_xOffset = 0.0f;
float InputHandler::s_yOffset = 0.0f;
float InputHandler::s_scrollOffset = 0.0f;

bool InputHandler::s_firstMouse = true;
bool InputHandler::s_pausePressed = false;
bool InputHandler::s_modePressed = false;
bool InputHandler::s_rightMouseDown = false;

bool InputHandler::s_screenshotPressed = false;
bool InputHandler::s_recordPressed = false;
bool InputHandler::s_overlayPressed = false;
bool InputHandler::s_trailsPressed = false;
bool InputHandler::s_bloomPressed = false;
bool InputHandler::s_evolutionPressed = false;
bool InputHandler::s_volumetricPressed = false;
bool InputHandler::s_fullscreenPressed = false;

int InputHandler::s_scenarioSwitch = -1;

InputHandler::InputHandler() : m_shouldClose(false) {}

void InputHandler::init(GLFWwindow* window) {
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}

void InputHandler::processInput(GLFWwindow* window, Camera& camera, float deltaTime) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        m_shouldClose = true;
    }

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) camera.processKeyboard(0, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) camera.processKeyboard(1, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) camera.processKeyboard(2, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) camera.processKeyboard(3, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) camera.processKeyboard(4, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) camera.processKeyboard(5, deltaTime);

    s_rightMouseDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

    if (s_rightMouseDown) {
        camera.processMouseMovement(s_xOffset, s_yOffset);
    }

    s_xOffset = 0.0f;
    s_yOffset = 0.0f;

    if (s_scrollOffset != 0.0f) {
        camera.processMouseScroll(s_scrollOffset);
        s_scrollOffset = 0.0f;
    }
}

bool InputHandler::isPauseToggled() {
    if (s_pausePressed) { s_pausePressed = false; return true; }
    return false;
}
bool InputHandler::isModeToggled() {
    if (s_modePressed) { s_modePressed = false; return true; }
    return false;
}
bool InputHandler::isScreenshotRequested() {
    if (s_screenshotPressed) { s_screenshotPressed = false; return true; }
    return false;
}
bool InputHandler::isRecordToggled() {
    if (s_recordPressed) { s_recordPressed = false; return true; }
    return false;
}
bool InputHandler::isOverlayToggled() {
    if (s_overlayPressed) { s_overlayPressed = false; return true; }
    return false;
}
bool InputHandler::isTrailsToggled() {
    if (s_trailsPressed) { s_trailsPressed = false; return true; }
    return false;
}
bool InputHandler::isBloomToggled() {
    if (s_bloomPressed) { s_bloomPressed = false; return true; }
    return false;
}
bool InputHandler::isEvolutionToggled() {
    if (s_evolutionPressed) { s_evolutionPressed = false; return true; }
    return false;
}
bool InputHandler::isVolumetricToggled() {
    if (s_volumetricPressed) { s_volumetricPressed = false; return true; }
    return false;
}
bool InputHandler::isFullscreenToggled() {
    if (s_fullscreenPressed) { s_fullscreenPressed = false; return true; }
    return false;
}
int InputHandler::getScenarioSwitch() {
    int v = s_scenarioSwitch;
    s_scenarioSwitch = -1;
    return v;
}

void InputHandler::mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    float x = static_cast<float>(xpos);
    float y = static_cast<float>(ypos);

    if (s_firstMouse) {
        s_lastX = x;
        s_lastY = y;
        s_firstMouse = false;
    }

    s_xOffset = x - s_lastX;
    s_yOffset = s_lastY - y;
    s_lastX = x;
    s_lastY = y;
}

void InputHandler::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    s_scrollOffset = static_cast<float>(yoffset);
}

void InputHandler::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS) return;

    switch (key) {
        case GLFW_KEY_SPACE: s_pausePressed = true; break;
        case GLFW_KEY_TAB: s_modePressed = true; break;
        case GLFW_KEY_F2: s_screenshotPressed = true; break;
        case GLFW_KEY_F3: s_recordPressed = true; break;
        case GLFW_KEY_F4: s_overlayPressed = true; break;
        case GLFW_KEY_F11: s_fullscreenPressed = true; break;

        case GLFW_KEY_B: s_bloomPressed = true; break;
        case GLFW_KEY_T: s_trailsPressed = true; break;
        case GLFW_KEY_V: s_evolutionPressed = true; break;
        case GLFW_KEY_G: s_volumetricPressed = true; break;

        case GLFW_KEY_1: s_scenarioSwitch = 0; break;
        case GLFW_KEY_2: s_scenarioSwitch = 1; break;
        case GLFW_KEY_3: s_scenarioSwitch = 2; break;
        case GLFW_KEY_4: s_scenarioSwitch = 3; break;

        default: break;
    }
}