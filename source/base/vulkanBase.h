#pragma once

#include <memory>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include "VulkanCommon.h"
#include "VulkanDevice.h"
#include "VulkanSwapchain.h"

class VulkanBase {
protected:
    
    virtual void createInstance();
    virtual void createPhysicalDevice();
    virtual void createVmaAllocator();

    void createSurface();
    

public:
    bool enableValidationLayers = true;
    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };
    std::vector<const char*> enabledDeviceExtensions;
	std::vector<const char*> enabledInstanceExtensions;

    uint32_t windowWidth = 800, windowHeight = 600;
    VkSurfaceKHR surface;

    VmaAllocator allocator;

    GLFWwindow* window;

    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    vki::VulkanDevice vulkanDevice;
    VkDevice device;

    VkPhysicalDeviceProperties physicalDeviceProperties;
	VkPhysicalDeviceFeatures physicalDeviceFeatures;
	VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;
    VkPhysicalDeviceFeatures enabledFeatures{};

    vki::VulkanSwapchain vulkanSwapchain;
    std::vector<VkFramebuffer> swapChainFramebuffers;

    VkFormat depthFormat;
    struct {
        VkImage image;
        VmaAllocation allocation;
        VkImageView view;
    } depthStencil;

    VkRenderPass renderPass;

    virtual bool isPhysicalDeviceSuitable(VkPhysicalDevice);
    virtual void findDepthFormat();

    /** @brief create a default render pass*/
    virtual void createRenderpass();
    virtual void createFrameBuffers();

    virtual void initVulkan();
    virtual void createWindow();
    virtual void prepare();
    virtual void renderLoop();
    virtual void cleanUp();
};