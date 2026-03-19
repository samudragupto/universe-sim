#include "input_handler.h"

float InputHandler::s_lx=960, InputHandler::s_ly=540;
float InputHandler::s_xo=0, InputHandler::s_yo=0, InputHandler::s_so=0;
bool InputHandler::s_first=true, InputHandler::s_pause=false, InputHandler::s_mode=false, InputHandler::s_rmb=false;
bool InputHandler::s_shot=false, InputHandler::s_rec=false, InputHandler::s_overlay=false;
bool InputHandler::s_trails=false, InputHandler::s_bloom=false, InputHandler::s_evol=false, InputHandler::s_vol=false;
int InputHandler::s_scen=-1;

InputHandler::InputHandler() : m_close(false) {}

void InputHandler::init(GLFWwindow* w) {
    glfwSetCursorPosCallback(w, mouseCallback);
    glfwSetScrollCallback(w, scrollCallback);
    glfwSetKeyCallback(w, keyCallback);
}

void InputHandler::processInput(GLFWwindow* w, Camera& cam, float dt) {
    if (glfwGetKey(w, GLFW_KEY_ESCAPE)==GLFW_PRESS) m_close=true;
    if (glfwGetKey(w, GLFW_KEY_W)==GLFW_PRESS) cam.processKeyboard(0, dt);
    if (glfwGetKey(w, GLFW_KEY_S)==GLFW_PRESS) cam.processKeyboard(1, dt);
    if (glfwGetKey(w, GLFW_KEY_A)==GLFW_PRESS) cam.processKeyboard(2, dt);
    if (glfwGetKey(w, GLFW_KEY_D)==GLFW_PRESS) cam.processKeyboard(3, dt);
    if (glfwGetKey(w, GLFW_KEY_Q)==GLFW_PRESS) cam.processKeyboard(4, dt);
    if (glfwGetKey(w, GLFW_KEY_E)==GLFW_PRESS) cam.processKeyboard(5, dt);
    s_rmb = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS;
    if (s_rmb) cam.processMouseMovement(s_xo, s_yo);
    s_xo=s_yo=0;
    if (s_so!=0) { cam.processMouseScroll(s_so); s_so=0; }
}

bool InputHandler::isPauseToggled() { if(s_pause){s_pause=false;return true;}return false; }
bool InputHandler::isModeToggled() { if(s_mode){s_mode=false;return true;}return false; }
bool InputHandler::isScreenshotRequested() { if(s_shot){s_shot=false;return true;}return false; }
bool InputHandler::isRecordToggled() { if(s_rec){s_rec=false;return true;}return false; }
bool InputHandler::isOverlayToggled() { if(s_overlay){s_overlay=false;return true;}return false; }
bool InputHandler::isTrailsToggled() { if(s_trails){s_trails=false;return true;}return false; }
bool InputHandler::isBloomToggled() { if(s_bloom){s_bloom=false;return true;}return false; }
bool InputHandler::isEvolutionToggled() { if(s_evol){s_evol=false;return true;}return false; }
bool InputHandler::isVolumetricToggled() { if(s_vol){s_vol=false;return true;}return false; }
int InputHandler::getScenarioSwitch() { int v=s_scen; s_scen=-1; return v; }

void InputHandler::mouseCallback(GLFWwindow*, double x, double y) {
    float xp=(float)x, yp=(float)y;
    if(s_first){s_lx=xp;s_ly=yp;s_first=false;}
    s_xo=xp-s_lx; s_yo=s_ly-yp; s_lx=xp; s_ly=yp;
}

void InputHandler::scrollCallback(GLFWwindow*, double, double y) { s_so=(float)y; }

void InputHandler::keyCallback(GLFWwindow*, int key, int, int act, int) {
    if(act!=GLFW_PRESS)return;
    switch(key) {
        case GLFW_KEY_SPACE: s_pause=true; break;
        case GLFW_KEY_TAB: s_mode=true; break;
        case GLFW_KEY_F2: s_shot=true; break;
        case GLFW_KEY_F3: s_rec=true; break;
        case GLFW_KEY_F4: s_overlay=true; break;
        case GLFW_KEY_T: s_trails=true; break;
        case GLFW_KEY_B: s_bloom=true; break;
        case GLFW_KEY_V: s_evol=true; break;
        case GLFW_KEY_G: s_vol=true; break;
        case GLFW_KEY_1: s_scen=0; break;
        case GLFW_KEY_2: s_scen=1; break;
        case GLFW_KEY_3: s_scen=2; break;
        case GLFW_KEY_4: s_scen=3; break;
    }
}