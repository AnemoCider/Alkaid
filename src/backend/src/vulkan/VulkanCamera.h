#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/matrix_transform.hpp>
#include <glm/glm.hpp>
#include <iostream>
#include "VulkanCommon.h"

namespace vki {
    class Camera {
    public:
        float deltaTime = 0.0f;	// Time between current frame and last frame
        float lastFrame = 0.0f; // Time of last frame
        Camera() = default;
        Camera(float xPos, float yPos, float zPos);

        glm::vec3 position = { 0.0f, 0.0f, 0.0f };
        glm::vec3 up = { 0.0f, 1.0f, 0.0f };
        glm::vec3 front = { 1.0f, 0.0f, 0.0f };
        glm::vec3 right;
        float yaw;
        float pitch;
        float moveSpeed = 2.5f;
        float sensitivity = 0.1f;
        float zoom = 45.0f;
        glm::mat4 viewMatrix = glm::mat4(1.0f);
        glm::mat4 projectionMatrix = glm::mat4(1.0f);

        // User is dragging the cursor
        bool inDrag = false;
        float lastX = 400, lastY = 300;
        bool firstMouse = true;

        glm::mat4 view();

        glm::mat4 projection(float width, float height);

        glm::mat4 projection(float width, float height, float zNear, float zFar);

        void move(GLFWwindow* window);

        void update(GLFWwindow* window);

        void zoomIn(float scrollOffset);

        void startDrag();

        void disableDrag();

        void mouseDrag(float xpos, float ypos);
    };
};