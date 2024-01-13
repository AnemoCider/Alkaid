#pragma once

#include "VulkanDevice.h"

class VulkanBase {
protected:
    
    virtual void createInstance();
    virtual void createPhysicalDevice();
    

public:
    bool enableValidationLayers = true;
    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };

    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;

    VkPhysicalDeviceProperties physicalDeviceProperties{};
	VkPhysicalDeviceFeatures physicalDeviceFeatures{};
	VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties{};

    virtual bool isPhysicalDeviceSuitable(VkPhysicalDevice);

    virtual void initVulkan();
    virtual void createWindow();
    virtual void prepare();
    virtual void renderLoop();
};