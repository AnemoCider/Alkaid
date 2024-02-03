#include "preparation/VulkanSwapChain.h"


using vki::SwapChain;

void vki::SwapChain::getSurfaceSupports() {
    supports.capabilities = instance->phyDevice.getSurfaceCapabilitiesKHR(instance->surface);
    supports.formats = instance->phyDevice.getSurfaceFormatsKHR(instance->surface);
    supports.presentModes = instance->phyDevice.getSurfacePresentModesKHR(instance->surface);
}

void vki::SwapChain::setUp() {
    for (const auto& availableFormat : supports.formats) {
        if (availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
            availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            setting.surfaceFormat = availableFormat;
            break;
        }
    }
    
    setting.presentMode = vk::PresentModeKHR::eFifo;
    for (const auto& availablePresentMode : supports.presentModes) {
        if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
            setting.presentMode = availablePresentMode;
            break;
        }
    }

    setting.imageCount = supports.capabilities.minImageCount + 1;
    if (supports.capabilities.maxImageCount > 0 && setting.imageCount > supports.capabilities.maxImageCount) {
        setting.imageCount = supports.capabilities.maxImageCount;
    }
    
    setting.extent = { instance->width, instance->height };

    if (supports.capabilities.currentExtent.width != (std::numeric_limits<uint32_t>::max)()) {
        setting.extent = supports.capabilities.currentExtent;
    } else {
        int width, height;
        glfwGetFramebufferSize(instance->window, &width, &height);

        VkExtent2D actualExtent = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };

        actualExtent.width = std::clamp(actualExtent.width, supports.capabilities.minImageExtent.width, supports.capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, supports.capabilities.minImageExtent.height, supports.capabilities.maxImageExtent.height);

        setting.extent = actualExtent;
    }
}

void vki::SwapChain::setInstance(vki::Instance* inst) {
    instance = inst;
}

void vki::SwapChain::setDevice(vki::Device* device) {
    this->device = device;
}

void vki::SwapChain::init() {
    getSurfaceSupports();
    setUp();
    vk::SwapchainCreateInfoKHR createInfo{
        .surface = instance->surface,
        .minImageCount = setting.imageCount,
        .imageFormat = setting.surfaceFormat.format,
        .imageColorSpace = setting.surfaceFormat.colorSpace,
        .imageExtent = setting.extent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
        .preTransform = supports.capabilities.currentTransform,
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode = setting.presentMode,
        .clipped = vk::True,
        .oldSwapchain = swapChain
    };
    swapChain = device->getDevice().createSwapchainKHR(createInfo);
}

void vki::SwapChain::clear() {
    device->getDevice().destroySwapchainKHR(swapChain);
}
