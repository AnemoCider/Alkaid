#include "control/VulkanCamera.h"

namespace vki {
	Camera::Camera(float xPos, float yPos, float zPos) : viewMatrix(glm::mat4(1.0f)), projectionMatrix(glm::mat4(1.0f)),
        position(xPos, yPos, zPos), front(1.0f, 0.0f, 0.0f), up(0.0f, 1.0f, 0.0f), yaw(0.0f), pitch(0.0f),
        moveSpeed(2.5f), sensitivity(0.1f), zoom(45.0f) {
        right = glm::cross(front, up);
    }

    glm::mat4 Camera::view() {
        viewMatrix = glm::lookAt(position, position + front, up);
        // viewMatrix = glm::lookAt(glm::vec3(4.0f, -4.0f, 1.5f), glm::vec3(0.0f, 0.0f, 1.5f), glm::vec3(0.0f, 0.0f, 1.0f));
        return viewMatrix;
    }

    glm::mat4 Camera::projection(float width, float height) {
        projectionMatrix = glm::perspective(glm::radians(zoom), (float)width / (float)height, 0.1f, 50.0f);
        return projectionMatrix;
    }

    glm::mat4 Camera::projection(float width, float height, float zNear, float zFar) {
        projectionMatrix = glm::perspective(glm::radians(zoom), (float)width / (float)height, zNear, zFar);
        return projectionMatrix;
    }

    void Camera::move(GLFWwindow* window) {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        glm::vec3 moveDir{ 0.0f, 0.0f, 0.0f };
        bool moved = false;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            moveDir += front;
            moved = true;
        } else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            moveDir -= front;
            moved = true;
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            moveDir -= right;
            moved = true;
        } else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            moveDir += right;
            moved = true;
        }
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
            moveDir += up;
            moved = true;
        } else if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
            moveDir -= up;
            moved = true;
        }
        if (moved)
            position += glm::normalize(moveDir) * moveSpeed * deltaTime;
    }

    void Camera::update(GLFWwindow* window) {
        // -1, 1, 0
        front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        front.y = sin(glm::radians(pitch));
        front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        front = glm::normalize(front);
        right = glm::cross(front, up);
        move(window);
    }

    void Camera::zoomIn(float scrollOffset) {
        zoom -= scrollOffset * 5.0f;
        if (zoom < 20.0f) zoom = 20.0f;
        if (zoom > 90.0f) zoom = 90.0f;
    }

    void Camera::startDrag() {
        inDrag = true;
    }

    void Camera::disableDrag() {
        inDrag = false;
        firstMouse = true;
    }

    void Camera::mouseDrag(float xpos, float ypos) {
        if (!inDrag) return;
        if (firstMouse) // initially set to true
        {
            lastX = xpos;
            lastY = ypos;
            firstMouse = false;
        }

        float xoffset = xpos - lastX;
        float yoffset = lastY - ypos; // reversed since y-coordinates range from bottom to top

        lastX = xpos;
        lastY = ypos;

        xoffset *= sensitivity;
        yoffset *= sensitivity;

        yaw += xoffset;
        pitch += yoffset;
        if (pitch > 89.0f)
            pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;
    }
};