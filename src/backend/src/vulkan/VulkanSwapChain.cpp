#include "preparation/VulkanSwapChain.h"

using vki::SwapChain;

vk::Extent2D vki::SwapChain::chooseExtent() {
    instance->supports.capabilities = instance->phyDevice.getSurfaceCapabilitiesKHR(instance->surface);
    if (instance->supports.capabilities.currentExtent.width != (std::numeric_limits<uint32_t>::max)()) {
        instance->width = instance->supports.capabilities.currentExtent.width;
        instance->height = instance->supports.capabilities.currentExtent.height;
        return instance->supports.capabilities.currentExtent;
    } else {
        int width, height;
        glfwGetFramebufferSize(instance->window, &width, &height);

        vk::Extent2D actualExtent = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };

        actualExtent.width = std::clamp(actualExtent.width, instance->supports.capabilities.minImageExtent.width, instance->supports.capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, instance->supports.capabilities.minImageExtent.height, instance->supports.capabilities.maxImageExtent.height);

        instance->width = actualExtent.width;
        instance->height = actualExtent.height;

        return actualExtent;
    }
}

void vki::SwapChain::setUp() {
    for (const auto& availableFormat : instance->supports.formats) {
        if (availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
            availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            setting.surfaceFormat = availableFormat;
            break;
        }
    }
    
    
    setting.presentMode = vk::PresentModeKHR::eFifo;
    for (const auto& availablePresentMode : instance->supports.presentModes) {
        if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
            setting.presentMode = availablePresentMode;
            break;
        }
    }

    setting.imageCount = instance->supports.capabilities.minImageCount + 1;
    if (instance->supports.capabilities.maxImageCount > 0 && setting.imageCount > instance->supports.capabilities.maxImageCount) {
        setting.imageCount = instance->supports.capabilities.maxImageCount;
    }
    
    setting.extent = chooseExtent();

    if (instance->supports.capabilities.currentExtent.width != (std::numeric_limits<uint32_t>::max)()) {
        setting.extent = instance->supports.capabilities.currentExtent;
    } else {
        int width, height;
        glfwGetFramebufferSize(instance->window, &width, &height);

        VkExtent2D actualExtent = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };

        actualExtent.width = std::clamp(actualExtent.width, instance->supports.capabilities.minImageExtent.width, instance->supports.capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, instance->supports.capabilities.minImageExtent.height, instance->supports.capabilities.maxImageExtent.height);

        setting.extent = actualExtent;
    }
}

void vki::SwapChain::setInstance(vki::Instance* inst) {
    instance = inst;
}

void vki::SwapChain::setDevice(vki::Device* device) {
    this->device = device;
}

vk::SwapchainKHR vki::SwapChain::init() {
    setUp();
    auto res = swapChain;
    vk::SwapchainCreateInfoKHR createInfo{
        .surface = instance->surface,
        .minImageCount = setting.imageCount,
        .imageFormat = setting.surfaceFormat.format,
        .imageColorSpace = setting.surfaceFormat.colorSpace,
        .imageExtent = setting.extent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
        .preTransform = instance->supports.capabilities.currentTransform,
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode = setting.presentMode,
        .clipped = vk::True,
        .oldSwapchain = swapChain
    };
    swapChain = device->getDevice().createSwapchainKHR(createInfo);
    createViews();
    return res;
}

void vki::SwapChain::clear() {
    for (auto& i : views) {
        device->getDevice().destroyImageView(i);
    }
    device->getDevice().destroySwapchainKHR(swapChain);
}

uint32_t vki::SwapChain::getImageCount() const {
    return setting.imageCount;
}

void vki::SwapChain::createViews() {
    std::vector<vk::Image> swapChainImages = device->getDevice().getSwapchainImagesKHR(swapChain);
    views.resize(swapChainImages.size());

    vk::ImageSubresourceRange subRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
    };
    for (uint32_t i = 0; i < swapChainImages.size(); i++) {
        
        vk::ImageViewCreateInfo viewCI{
            .image = swapChainImages[i],
            .viewType = vk::ImageViewType::e2D,
            .format = setting.surfaceFormat.format,
            .subresourceRange = subRange
        };
        views[i] = device->getDevice().createImageView(viewCI, nullptr);
    }
}

vk::Format vki::SwapChain::getColorFormat() const {
    return setting.surfaceFormat.format;
}

vk::SwapchainKHR& vki::SwapChain::getSwapChain(){
    return this->swapChain;
}
