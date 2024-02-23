#include "VulkanDevice.h"
#include <stdexcept>

using vki::Device;

void vki::Device::checkExtensions() {
    std::vector<vk::ExtensionProperties> availableExtensions = instance->phyDevice.enumerateDeviceExtensionProperties();
    for (const char* deviceExtension : deviceExtensions) {
        if (std::none_of(availableExtensions.begin(), availableExtensions.end(),
            [deviceExtension](const vk::ExtensionProperties& extension) {
                return strcmp(deviceExtension, extension.extensionName) == 0;
            })) {
            throw std::runtime_error("Device extension not supported: " + std::string(deviceExtension));
        }
    }
}

void vki::Device::setInstance(vki::Instance* inst) {
    instance = inst;
}


void Device::init() {
    checkExtensions();

    float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo queueCreateInfo{
        .flags = vk::DeviceQueueCreateFlags(),
        .queueFamilyIndex = instance->grqFamilyIndex,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority
    };

    // enable sampler anisotropy for image sampler
    vk::PhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.samplerAnisotropy = vk::True;

    auto createInfo = vk::DeviceCreateInfo{
        .flags = vk::DeviceCreateFlags(),
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queueCreateInfo,
        .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data(),
        .pEnabledFeatures = &deviceFeatures
    };

    device = instance->phyDevice.createDevice(createInfo);
}

void vki::Device::clear() {
    device.destroy();
}

void vki::Device::addExtension(const char* ext) {
    deviceExtensions.push_back(ext);
}

void vki::Device::getGraphicsQueue(vk::Queue& queue) {
    queue = device.getQueue(getGrqFamilyIndex(), 0);
}

vk::Device vki::Device::getDevice() {
    return device;
}

uint32_t vki::Device::getGrqFamilyIndex() {
    return instance->grqFamilyIndex;
}

uint32_t vki::Device::getMemoryType(uint32_t typeBits, vk::MemoryPropertyFlags props) {
    for (uint32_t i = 0; i < instance->supports.memProperties.memoryTypeCount; i++) {
        if ((typeBits & 1) == 1) {
            if ((instance->supports.memProperties.memoryTypes[i].propertyFlags & props) == props) {
                return i;
            }
        }
        typeBits >>= 1;
    }
    std::runtime_error("No proper memory type found.\n");
    return 0;
}
