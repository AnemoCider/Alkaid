#include "vulkanBase.h"

#define VMA_IMPLEMENTATION
#include <vulkan/vk_mem_alloc.h>
#include <stdexcept>
#include <iostream>
#include <vector>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

void VulkanBase::initVulkan() {
    createInstance();
}

void VulkanBase::createInstance() {
    // Optional, info about app
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; // type of struct
    appInfo.pNext = { nullptr }; // pointer to extension info in the future
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    // Required, specify global extensions and validation layers to use
    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;

    // We use GLFW here, so we ask GLFW what extensions it requires
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    if (!glfwExtensions)
        std::cout << "Error occurred when getting required extensions.\n";

    createInfo.enabledExtensionCount = glfwExtensionCount;
    createInfo.ppEnabledExtensionNames = glfwExtensions; // use glfw ext.

    // check extension support
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());
    std::cout << "available instance level extensions:\n";

    for (const auto& extension : extensions) {
        std::cout << '\t' << extension.extensionName << '\n';
    }

    std::cout << "enabled instance level extensions:\n";

    for (unsigned int i = 0; i < createInfo.enabledExtensionCount; i++) {
        std::cout << '\t' << createInfo.ppEnabledExtensionNames[i] << '\n';
    }

    createInfo.enabledLayerCount = 0;
    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    }
    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("failed to create instance!");
    }
}

void VulkanBase::createPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
    for (const auto& device : devices) {
        if (isPhysicalDeviceSuitable(device)) {
            physicalDevice = device;
            break;
        }
    }

    if (physicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
    
    vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);
	vkGetPhysicalDeviceFeatures(physicalDevice, &physicalDeviceFeatures);
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &physicalDeviceMemoryProperties);
}

// Return true if the device is suitable
// This function should be overriden
bool VulkanBase::isPhysicalDeviceSuitable(VkPhysicalDevice physicalDevice) {
    return true;
}

void VulkanBase::createWindow() {
    glfwInit();
}
void VulkanBase::prepare() {}
void VulkanBase::renderLoop() {}


