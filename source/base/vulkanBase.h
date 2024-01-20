#pragma once

#include <memory>
#include <vector>
#include <string>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include "VulkanCommon.h"
#include "VulkanDevice.h"
#include "VulkanSwapchain.h"
#include "VulkanInit.h"

class VulkanBase {
protected:
    bool prepared = false;

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
    VkQueue graphicsQueue;

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

    VkRenderPass renderPass { VK_NULL_HANDLE };

    VkCommandPool commandPool { VK_NULL_HANDLE };
    std::vector<VkCommandBuffer> commandBuffers;

    virtual std::string getShaderPathName() = 0;
    std::vector<char> readFile(const std::string& filename);
    VkShaderModule createShaderModule(const std::vector<char>& code);

    virtual bool isPhysicalDeviceSuitable(VkPhysicalDevice);
    virtual void findDepthFormat();

    virtual void createDepthStencil();
    /** @brief create a default render pass*/
    virtual void createRenderpass();
    virtual void createFrameBuffers();

    virtual void createCommandBuffers() = 0;

    virtual void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, 
    VmaAllocationCreateFlags flags, VkBuffer& buffer, VmaAllocation& allocation, VmaAllocationInfo* pAllocInfo = nullptr);
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);
    virtual void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize sz);

    virtual void createImage(VkImageCreateInfo imageCreateInfo, VkImage& image, VmaAllocation& alloc);

    virtual void render() = 0;
    virtual void recreateSwapChain();

    virtual void addExtensions(const std::vector<const char*>& instanceExt, const std::vector<const char*>& deviceExt);

    virtual void initVulkan();
    virtual void createWindow();
    virtual void prepare();
    virtual void renderLoop();
    virtual void cleanUp();
};