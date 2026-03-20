#include "camera.h"
#include <algorithm>
#include <cmath>

Camera::Camera()
    : m_position(0.0f, 0.0f, 120.0f)
    , m_target(0.0f, 0.0f, 0.0f)
    , m_front(0.0f, 0.0f, -1.0f)
    , m_up(0.0f, 1.0f, 0.0f)
    , m_right(1.0f, 0.0f, 0.0f)
    , m_worldUp(0.0f, 1.0f, 0.0f)
    , m_yaw(-90.0f)
    , m_pitch(0.0f)
    , m_fov(60.0f)
    , m_aspect(16.0f / 9.0f)
    , m_near(0.01f)
    , m_far(50000.0f)
    , m_speed(25.0f)
    , m_sensitivity(0.1f)
    , m_orbitDistance(120.0f)
    , m_mode(CameraMode::FREE)
    , m_currentKeyframe(0)
    , m_keyframeTime(0.0f)
    , m_cinematicDone(false) {
    updateVectors();
}

void Camera::setPosition(const glm::vec3& pos) { m_position = pos; }
void Camera::setTarget(const glm::vec3& target) { m_target = target; }
void Camera::setFOV(float fov) { m_fov = fov; }
void Camera::setAspectRatio(float aspect) { m_aspect = aspect; }
void Camera::setNearFar(float nearP, float farP) { m_near = nearP; m_far = farP; }
void Camera::setSpeed(float speed) { m_speed = speed; }
void Camera::setSensitivity(float sens) { m_sensitivity = sens; }

void Camera::processMouseMovement(float xoffset, float yoffset) {
    if (m_mode == CameraMode::CINEMATIC) return;

    xoffset *= m_sensitivity;
    yoffset *= m_sensitivity;

    if (m_mode == CameraMode::FREE) {
        m_yaw += xoffset;
        m_pitch += yoffset;
        m_pitch = std::clamp(m_pitch, -89.0f, 89.0f);
        updateVectors();
    } else if (m_mode == CameraMode::ORBIT) {
        m_yaw += xoffset;
        m_pitch += yoffset;
        m_pitch = std::clamp(m_pitch, -89.0f, 89.0f);

        float ry = glm::radians(m_yaw);
        float rp = glm::radians(m_pitch);

        m_position.x = m_target.x + m_orbitDistance * cosf(rp) * cosf(ry);
        m_position.y = m_target.y + m_orbitDistance * sinf(rp);
        m_position.z = m_target.z + m_orbitDistance * cosf(rp) * sinf(ry);

        m_front = glm::normalize(m_target - m_position);
        m_right = glm::normalize(glm::cross(m_front, m_worldUp));
        m_up = glm::normalize(glm::cross(m_right, m_front));
    }
}

void Camera::processMouseScroll(float yoffset) {
    if (m_mode == CameraMode::CINEMATIC) return;

    if (m_mode == CameraMode::ORBIT) {
        m_orbitDistance -= yoffset * m_orbitDistance * 0.1f;
        m_orbitDistance = std::max(5.0f, m_orbitDistance);

        float ry = glm::radians(m_yaw);
        float rp = glm::radians(m_pitch);

        m_position.x = m_target.x + m_orbitDistance * cosf(rp) * cosf(ry);
        m_position.y = m_target.y + m_orbitDistance * sinf(rp);
        m_position.z = m_target.z + m_orbitDistance * cosf(rp) * sinf(ry);
    } else {
        float scaled = m_speed * std::max(1.0f, getDistance() * 0.03f);
        m_position += m_front * yoffset * scaled * 0.5f;
    }
}

void Camera::processKeyboard(int direction, float dt) {
    if (m_mode == CameraMode::CINEMATIC) return;

    float speed = m_speed * dt * std::max(1.0f, getDistance() * 0.03f);

    switch (direction) {
        case 0: m_position += m_front * speed; break;
        case 1: m_position -= m_front * speed; break;
        case 2: m_position -= m_right * speed; break;
        case 3: m_position += m_right * speed; break;
        case 4: m_position += m_worldUp * speed; break;
        case 5: m_position -= m_worldUp * speed; break;
    }
}

void Camera::setMode(CameraMode mode) {
    m_mode = mode;
    if (mode == CameraMode::ORBIT) {
        m_target = glm::vec3(0.0f, 0.0f, 0.0f);
        m_orbitDistance = glm::length(m_position - m_target);
        if (m_orbitDistance < 5.0f) m_orbitDistance = 120.0f;
    } else if (mode == CameraMode::CINEMATIC) {
        setupCinematicTour();
    }
}

glm::mat4 Camera::getViewMatrix() const {
    if (m_mode == CameraMode::ORBIT || m_mode == CameraMode::CINEMATIC) {
        return glm::lookAt(m_position, m_target, m_worldUp);
    }
    return glm::lookAt(m_position, m_position + m_front, m_up);
}

glm::mat4 Camera::getProjectionMatrix() const {
    return glm::perspective(glm::radians(m_fov), m_aspect, m_near, m_far);
}

float Camera::getDistance() const {
    return glm::length(m_position);
}

void Camera::update(float dt) {
    if (m_mode == CameraMode::CINEMATIC) updateCinematic(dt);
}

void Camera::setupCinematicTour() {
    m_keyframes.clear();
    m_currentKeyframe = 0;
    m_keyframeTime = 0.0f;
    m_cinematicDone = false;

    m_keyframes.push_back({glm::vec3(0, 0, 120), glm::vec3(0, 0, 0), 6.0f, 60.0f});
    m_keyframes.push_back({glm::vec3(0, 20, 90), glm::vec3(0, 0, 0), 6.0f, 55.0f});
    m_keyframes.push_back({glm::vec3(40, 10, 70), glm::vec3(0, 0, 0), 6.0f, 50.0f});
    m_keyframes.push_back({glm::vec3(-40, 10, 70), glm::vec3(0, 0, 0), 6.0f, 50.0f});
}

void Camera::updateCinematic(float dt) {
    if (m_cinematicDone || m_keyframes.empty()) return;
    if (m_currentKeyframe >= (int)m_keyframes.size() - 1) {
        m_cinematicDone = true;
        return;
    }

    m_keyframeTime += dt;

    const auto& a = m_keyframes[m_currentKeyframe];
    const auto& b = m_keyframes[m_currentKeyframe + 1];

    float t = m_keyframeTime / a.duration;
    if (t >= 1.0f) {
        m_currentKeyframe++;
        m_keyframeTime = 0.0f;
        if (m_currentKeyframe >= (int)m_keyframes.size() - 1) {
            m_cinematicDone = true;
            return;
        }
        t = 0.0f;
    }

    float s = t * t * (3.0f - 2.0f * t);
    m_position = glm::mix(a.position, b.position, s);
    m_target = glm::mix(a.target, b.target, s);
    m_fov = glm::mix(a.fov, b.fov, s);
}

void Camera::updateVectors() {
    glm::vec3 front;
    front.x = cos(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
    front.y = sin(glm::radians(m_pitch));
    front.z = sin(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));

    m_front = glm::normalize(front);
    m_right = glm::normalize(glm::cross(m_front, m_worldUp));
    m_up = glm::normalize(glm::cross(m_right, m_front));
}