#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

enum class CameraMode {
    FREE,
    ORBIT,
    OVERVIEW,
    CINEMATIC
};

struct CinematicKeyframe {
    glm::vec3 position;
    glm::vec3 target;
    float duration;
    float fov;
};

class Camera {
public:
    Camera();

    void setPosition(const glm::vec3& pos);
    void setTarget(const glm::vec3& target);
    void setFOV(float fov);
    void setAspectRatio(float aspect);
    void setNearFar(float near, float far);
    void setSpeed(float speed);
    void setSensitivity(float sens);

    void processMouseMovement(float xoffset, float yoffset);
    void processMouseScroll(float yoffset);
    void processKeyboard(int direction, float deltaTime);

    void setMode(CameraMode mode);
    CameraMode getMode() const { return m_mode; }

    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix() const;
    glm::vec3 getPosition() const { return m_position; }
    float getDistance() const;

    void update(float dt);

    void setupCinematicTour();
    bool isCinematicDone() const { return m_cinematicDone; }

private:
    void updateVectors();
    void updateCinematic(float dt);

    glm::vec3 m_position;
    glm::vec3 m_target;
    glm::vec3 m_front;
    glm::vec3 m_up;
    glm::vec3 m_right;
    glm::vec3 m_worldUp;

    float m_yaw;
    float m_pitch;
    float m_fov;
    float m_aspect;
    float m_near;
    float m_far;
    float m_speed;
    float m_sensitivity;
    float m_orbitDistance;

    CameraMode m_mode;

    std::vector<CinematicKeyframe> m_keyframes;
    int m_currentKeyframe;
    float m_keyframeTime;
    bool m_cinematicDone;
};