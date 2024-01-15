#include <vector>
#include <stdexcept>
#include <unordered_set>

#include "VulkanDevice.h"
#include "VulkanInit.h"

using vki::VulkanDevice;

VulkanDevice::VulkanDevice(VkPhysicalDevice physicalDevice) : physicalDevice(physicalDevice) {}

void VulkanDevice::getQueueFamilyIndices(VkQueueFlags requestedQueueTypes) {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    uint32_t i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (requestedQueueTypes & VK_QUEUE_GRAPHICS_BIT & queueFamily.queueFlags) {
            queueFamilyIndices.graphics = i;
        } 
        if (requestedQueueTypes & VK_QUEUE_TRANSFER_BIT & queueFamily.queueFlags) {
            queueFamilyIndices.transfer = i;
        } 
        if (requestedQueueTypes & VK_QUEUE_COMPUTE_BIT& queueFamily.queueFlags) {
            queueFamilyIndices.compute = i;
        }
        i++;
    }
}

void VulkanDevice::checkDeviceExtensionSupport(const std::vector<const char *> & enabledExtensions) const {
    // Check for device extensions support
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, availableExtensions.data());
    std::unordered_set<std::string> requiredExtensions(enabledExtensions.begin(), enabledExtensions.end());
    for (const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }
    if (!requiredExtensions.empty())
        throw std::runtime_error("some requested logical device extensions not supported!");
}

void VulkanDevice::createLogicalDevice(VkPhysicalDeviceFeatures enabledFeatures, 
    std::vector<const char *> enabledExtensions, void *pNextChain, 
    bool useSwapChain, VkQueueFlags requestedQueueTypes) {
    
    getQueueFamilyIndices(requestedQueueTypes);

    std::vector<const char*> deviceExtensions(enabledExtensions);
    if (useSwapChain)
    {
        deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    checkDeviceExtensionSupport(enabledExtensions);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

    if (queueFamilyIndices.graphics.has_value()) {
        auto graphicsQueueInfo = vki::device_queue_create_info(queueFamilyIndices.graphics.value());
        queueCreateInfos.push_back(graphicsQueueInfo);
    }
    if (queueFamilyIndices.transfer.has_value()) {
        auto transferQueueInfo = vki::device_queue_create_info(queueFamilyIndices.transfer.value());
        queueCreateInfos.push_back(transferQueueInfo);
    }
    if (queueFamilyIndices.compute.has_value()) {
        auto computeQueueInfo = vki::device_queue_create_info(queueFamilyIndices.compute.value());
        queueCreateInfos.push_back(computeQueueInfo);
    }

    /* 
        TODO: support for other queue families
    */

    enabledFeatures.samplerAnisotropy = VK_TRUE;
    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();

    createInfo.pEnabledFeatures = &enabledFeatures;

    // Enable device-level extensions
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    VK_CHECK(vkCreateDevice(physicalDevice, &createInfo, nullptr, &logicalDevice));
}

void VulkanDevice::cleanUp() {
    vkDestroyDevice(logicalDevice, nullptr);
}
