#pragma once

#include <exception>
#include <vector>

#include <VulkanCommon.h>

namespace vki {
class VulkanDevice {
private:
    void getQueueFamilyIndices(VkQueueFlags requestedQueueTypes);
    void checkDeviceExtensionSupport(const std::vector<const char *> & enabledExtensions) const;

public:
    VulkanDevice() = default;
    VulkanDevice(VkPhysicalDevice physicalDevice);

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice logicalDevice = VK_NULL_HANDLE;
    
    QueueFamilyIndices queueFamilyIndices;

    void createLogicalDevice(VkPhysicalDeviceFeatures enabledFeatures, std::vector<const char *> enabledExtensions, void *pNextChain, bool useSwapChain = true, VkQueueFlags requestedQueueTypes = VK_QUEUE_GRAPHICS_BIT);
    void cleanUp();
};
};