#include "control/VulkanCamera.h"

namespace vki {
	Camera::Camera(float xPos, float yPos, float zPos) : viewMatrix(glm::mat4(1.0f)), projectionMatrix(glm::mat4(1.0f)),
        position(xPos, yPos, zPos), up(0.0f, 1.0f, 0.0f), yaw(180.0f), pitch(0.0f),
        moveSpeed(2.5f), sensitivity(0.1f), zoom(45.0f) {
        right = glm::cross(front, up);
    }
};