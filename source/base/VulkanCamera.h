#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/matrix_transform.hpp>
#include <glm/glm.hpp>
#include <iostream>

namespace Merak {
    class Camera {
    public:
        Camera(float xPos, float yPos, float zPos) : viewMatrix(glm::mat4(1.0f)), projectionMatrix(glm::mat4(1.0f)),
            position(xPos, yPos, zPos), up(0.0f, 0.0f, 1.0f), yaw(180.0f), pitch(45.0f),
            moveSpeed(0.001f), mouseSensitivity(0.1f), zoom(45.0f) {
            right = glm::cross(front, up);
        }

        glm::vec3 position;
        glm::vec3 up;
        glm::vec3 front;
        glm::vec3 right;
        float yaw;
        float pitch;
        float moveSpeed;
        float mouseSensitivity;
        float zoom;
        glm::mat4 viewMatrix;
        glm::mat4 projectionMatrix;

        decltype(auto) view() {
            viewMatrix = glm::lookAt(position, position + front, up);
            // viewMatrix = glm::lookAt(glm::vec3(4.0f, -4.0f, 1.5f), glm::vec3(0.0f, 0.0f, 1.5f), glm::vec3(0.0f, 0.0f, 1.0f));
            return viewMatrix;
        }

        decltype(auto) projection(float width, float height) {
            projectionMatrix = glm::perspective(glm::radians(zoom), (float)width / (float)height, 0.1f, 10.0f);
            return projectionMatrix;
        }

        void move(GLFWwindow* window) {
            glm::vec3 moveDir{ 0.0f, 0.0f, 0.0f };
            bool moved = false;
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
                moveDir += front;
                moved = true;
            }
            else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
                moveDir -= front;
                moved = true;
            }
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
                moveDir -= right;
                moved = true;
            }
            else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
                moveDir += right;
                moved = true;
            }
            if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
                moveDir += up;
                moved = true;
            }
            else if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
                moveDir -= up;
                moved = true;
            }
            if (moved)
                position += glm::normalize(moveDir) * moveSpeed;
        }

        decltype(auto) update(GLFWwindow* window) {
            // -1, 1, 0
            front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
            front.y = sin(glm::radians(pitch));
            front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
            front = glm::normalize(front);
            right = glm::cross(front, up);
            move(window);
        }
    };
};