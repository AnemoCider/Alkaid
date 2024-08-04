#include "VulkanDriver.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <string_view>
#include <vector>
#include <iostream>

namespace vki {

namespace {
/**
 * Returns the instance extensions required by the swap chain.
 */
std::vector<const char*> getSwapchainInstanceExtensions() {
    std::vector<const char*> ret;
    return ret;
}

std::vector<const char*> getInstanceExtensions() {
    std::vector<const char*> ret;

    uint32_t glfwExtCount;
    const char** pglfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    for (uint32_t i = 0; i < glfwExtCount; i++) {
        ret.emplace_back(pglfwExts[i]);
    }

    #ifdef __APPLE__
        extensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    #endif

    return ret;
}

vk::Instance createInstance() {
    std::string_view appName = "Example";
    #ifdef APP_NAME
        appName = APP_NAME;
    #endif

    vk::ApplicationInfo appInfo{
        .pApplicationName = appName.data(),
        .applicationVersion = 1,
        .pEngineName = "Alkaid",
        .engineVersion = 1,
        .apiVersion = VK_API_VERSION_1_3
    };

    std::vector<const char*> layers;
    if (VK_VALIDATION_ENABLED) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }

    std::vector<const char*> extensions = getSwapchainInstanceExtensions();
    std::vector<const char*> instExts = getInstanceExtensions();
    extensions.insert(extensions.end(), 
        std::make_move_iterator(instExts.begin()), 
        std::make_move_iterator(instExts.end()));

    vk::InstanceCreateInfo insInfo{
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = static_cast<uint32_t>(layers.size()),
        .ppEnabledLayerNames = layers.data(),
        .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data()
    };

    #ifdef __APPLE__
        insInfo.flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
    #endif
    return vk::createInstance(insInfo);
}

vk::PhysicalDevice selectPhysicalDevice(vk::Instance instance, uint32_t& graphicsQueueFamilyIndex) {
    std::vector<vk::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
    for (int i = 0; i < physicalDevices.size(); i++) {
        vk::PhysicalDeviceProperties properties = physicalDevices[i].getProperties();
        if (properties.apiVersion < VK_API_VERSION) continue;
        if (properties.deviceType != vk::PhysicalDeviceType::eDiscreteGpu) continue;
            
        std::vector<vk::QueueFamilyProperties> queueFamilies = physicalDevices[i].getQueueFamilyProperties();
        for (uint32_t j = 0; j < queueFamilies.size(); j++) {
            if (queueFamilies[j].queueFlags & vk::QueueFlagBits::eGraphics) {
                graphicsQueueFamilyIndex = j;
                return physicalDevices[i];
            }
        }
    }
    return VK_NULL_HANDLE;
}

std::vector<const char*> getDeviceExtensions(vk::PhysicalDevice physicalDevice) {
    std::vector<const char*> desiredExtensions, ret;

    #ifdef __APPLE__
        desiredExtensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
    #endif

    std::vector<vk::ExtensionProperties> supportedExtensions = physicalDevice.enumerateDeviceExtensionProperties();

    for (auto const& ext : desiredExtensions) {
        if (std::find_if(supportedExtensions.begin(), supportedExtensions.end(), 
            [ext](const vk::ExtensionProperties& supportedExt) {
                return strcmp(ext, supportedExt.extensionName) == 0;
            }) != supportedExtensions.end()) {
            ret.push_back(ext);
        }
    }

    return ret;
}

vk::Device createLogicalDevice(
    vk::PhysicalDevice physicalDevice, 
    vk::PhysicalDeviceFeatures deviceFeatures,
    uint32_t graphicsQueueFamilyIndex, 
    const std::vector<const char*>& deviceExtensions) {
    
    float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo queueCreateInfo{
        .flags = vk::DeviceQueueCreateFlags(),
        .queueFamilyIndex = graphicsQueueFamilyIndex,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority
    };

    // enable sampler anisotropy for image sampler
    vk::PhysicalDeviceFeatures enabledFeatures {
        .samplerAnisotropy = deviceFeatures.samplerAnisotropy
    };

    auto createInfo = vk::DeviceCreateInfo{
        .flags = vk::DeviceCreateFlags(),
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queueCreateInfo,
        .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data(),
        .pEnabledFeatures = &enabledFeatures
    };

    return physicalDevice.createDevice(createInfo);
}

} // namespace

Driver *Driver::create() {
    Driver* driver = new Driver();

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    driver->mInstance = createInstance();

    driver->mPhysicalDevice = selectPhysicalDevice(driver->mInstance, driver->mGraphicsQueueFamilyIndex);
    assert(driver->mPhysicalDevice != VK_NULL_HANDLE && "Failed to find a suitable GPU!");
    std::cout << "Selected physical device: " << driver->mPhysicalDevice.getProperties().deviceName << std::endl;
    driver->mContext.mPhysicalDeviceFeatures = driver->mPhysicalDevice.getFeatures();
    driver->mContext.mPhysicalDeviceProperties = driver->mPhysicalDevice.getProperties();
    driver->mContext.mPhysicalDeviceMemoryProperties = driver->mPhysicalDevice.getMemoryProperties();

    driver->mDevice = createLogicalDevice(
        driver->mPhysicalDevice, 
        driver->mContext.mPhysicalDeviceFeatures, 
        driver->mGraphicsQueueFamilyIndex, 
        getDeviceExtensions(driver->mPhysicalDevice));

    return driver;
}

Driver::Driver() = default;

Driver::~Driver() {
    mDevice.destroy();
    mInstance.destroy();
}

} // namespace vki