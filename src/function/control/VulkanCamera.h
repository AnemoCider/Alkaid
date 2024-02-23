#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/matrix_transform.hpp>
#include <glm/glm.hpp>
#include <iostream>
#include "common/VulkanCommon.h"

namespace vki {
    class Camera {
    public:
        float deltaTime = 0.0f;	// Time between current frame and last frame
        float lastFrame = 0.0f; // Time of last frame
        Camera(float xPos, float yPos, float zPos);

        glm::vec3 position;
        glm::vec3 up;
        glm::vec3 front;
        glm::vec3 right;
        float yaw;
        float pitch;
        float moveSpeed;
        float sensitivity;
        float zoom;
        glm::mat4 viewMatrix;
        glm::mat4 projectionMatrix;

        // User is dragging the cursor
        bool inDrag = false;
        float lastX = 400, lastY = 300;
        bool firstMouse = true;

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
            float currentFrame = glfwGetTime();
            deltaTime = currentFrame - lastFrame;
            lastFrame = currentFrame;
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
            else if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
                moveDir -= up;
                moved = true;
            }
            if (moved)
                position += glm::normalize(moveDir) * moveSpeed * deltaTime;
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

        void zoomIn(double scrollOffset) {
            zoom -= scrollOffset * 5.0f;
            if (zoom < 20.0f) zoom = 20.0f;
            if (zoom > 90.0f) zoom = 90.0f;
        }

        void startDrag() {
            inDrag = true;
        }

        void disableDrag() {
            inDrag = false;
            firstMouse = true;
        }

        void mouseDrag(double xpos, double ypos) {
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
};