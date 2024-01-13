#pragma once

#include <vulkan/vk_mem_alloc.h>
#include <exception>
#include <vector>

namespace vki {
class VulkanDevice {
private:
    virtual void getQueueFamilies();
public:
    VulkanDevice(VkPhysicalDevice physicalDevice);
    VkPhysicalDevice physicalDevice;
    VkDevice device;

    VkPhysicalDeviceFeatures deviceFeatures;
    const std::vector<const char*> deviceExtensions = {
        // Presenting images is NOT a vulkan core function.
        // We have to check and enable it at the device level.
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    struct {
        uint32_t graphics;
        uint32_t transfer;
        uint32_t compute;
    } queueFamilyIndices;

    // static virtual void createLogicalDevice();

};
};