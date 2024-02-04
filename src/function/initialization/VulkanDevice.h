#pragma once

#include <vector>
#include "initialization/VulkanInstance.h"


namespace vki {
class Device {
private:
    
    vk::Device device;
    vki::Instance* instance;
    std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };

    void checkExtensions();

public:
    
    void setInstance(vki::Instance* inst);
    

    /*
        Initialize logical device.
        Call this after setting desired properties.
    */
    void init();

    void clear();

    void addExtension(const char* ext);

    void getGraphicsQueue(vk::Queue& queue);

    vk::Device& getDevice();
    uint32_t getGrqFamilyIndex();

};
};