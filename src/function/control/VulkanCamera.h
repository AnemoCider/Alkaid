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
        Camera() = default;
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

        glm::mat4 view();

        glm::mat4 projection(float width, float height);

        void move(GLFWwindow* window);

        void update(GLFWwindow* window);

        void zoomIn(float scrollOffset);

        void startDrag();

        void disableDrag();

        void mouseDrag(float xpos, float ypos);
    };
};