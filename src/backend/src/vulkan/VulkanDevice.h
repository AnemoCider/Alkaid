#pragma once

#include <vector>
#include "VulkanInstance.h"


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

    vk::Device getDevice();
    uint32_t getGrqFamilyIndex();
    /**
     * @brief find a proper memory type index
     * @param typeBits memory type bits, typically obtained from memoryRequirements
     * @param props properties of the memory type requested, e.g., device local
     * @return index of the memory type from memoryTypes array of the physical device memory properties
    */
    uint32_t getMemoryType(uint32_t typeBits, vk::MemoryPropertyFlags props);
};
};