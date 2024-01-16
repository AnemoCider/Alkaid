#include "VulkanBase.h"

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <vector>
#include <array>


void VulkanBase::initVulkan() {
    createInstance();
    createPhysicalDevice();
    vulkanDevice = vki::VulkanDevice(physicalDevice);
    vulkanDevice.createLogicalDevice(enabledFeatures, enabledDeviceExtensions, nullptr);
    device = vulkanDevice.logicalDevice;
    vkGetDeviceQueue(device, vulkanDevice.queueFamilyIndices.graphics.value(), 0, &graphicsQueue);

    findDepthFormat();
    createSurface();
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
    enabledInstanceExtensions.insert(enabledInstanceExtensions.end(), 
        glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (!glfwExtensions)
        std::cout << "Error occurred when getting required extensions.\n";

    createInfo.enabledExtensionCount = enabledInstanceExtensions.size();
    createInfo.ppEnabledExtensionNames = enabledInstanceExtensions.data(); // use glfw ext.

    createInfo.enabledLayerCount = 0;
    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    }
    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance));

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

void VulkanBase::createVmaAllocator() {
    VmaVulkanFunctions vulkanFunctions = {};
    vulkanFunctions.vkGetInstanceProcAddr = &vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr = &vkGetDeviceProcAddr;
    
    VmaAllocatorCreateInfo allocatorCreateInfo = {};
    allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_2;
    allocatorCreateInfo.physicalDevice = physicalDevice;
    allocatorCreateInfo.device = device;
    allocatorCreateInfo.instance = instance;
    allocatorCreateInfo.pVulkanFunctions = &vulkanFunctions;
    
    vmaCreateAllocator(&allocatorCreateInfo, &allocator);
}


// Return true if the device is suitable
// This function should be overriden
bool VulkanBase::isPhysicalDeviceSuitable(VkPhysicalDevice physicalDevice) {
    return true;
}

void VulkanBase::createWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(windowWidth, windowHeight, "Vulkan", nullptr, nullptr);
    // glfwSetWindowUserPointer(window, this);
    if (!window) {
        std::cout << "Creating glfw window error!\n";
    }
}

void VulkanBase::createSurface() {
    VK_CHECK(glfwCreateWindowSurface(instance, window, nullptr, &surface));
}

void VulkanBase::findDepthFormat() {
    std::vector<VkFormat> formatList = {
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D24_UNORM_S8_UINT,
        VK_FORMAT_D16_UNORM_S8_UINT,
        VK_FORMAT_D16_UNORM
    };
    for (auto& format : formatList) {
        VkFormatProperties formatProps;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &formatProps);
        if (formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
        {
            depthFormat = format;
            break;
        }
    }
}

void VulkanBase::createDepthStencil() {
    VkImageCreateInfo imageInfo { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };

	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.format = depthFormat;
	imageInfo.extent = { windowWidth, windowHeight, 1 };
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VmaAllocationCreateInfo allocationCreateInfo {};
    allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

	VK_CHECK(vmaCreateImage(allocator, &imageInfo, &allocationCreateInfo, &depthStencil.image, &depthStencil.allocation, nullptr));

    VkImageViewCreateInfo depthStencilViewCI{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    depthStencilViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
    depthStencilViewCI.format = depthFormat;
    depthStencilViewCI.subresourceRange = {};
    depthStencilViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (depthFormat >= VK_FORMAT_D16_UNORM_S8_UINT) {
        depthStencilViewCI.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    depthStencilViewCI.subresourceRange.baseMipLevel = 0;
    depthStencilViewCI.subresourceRange.levelCount = 1;
    depthStencilViewCI.subresourceRange.baseArrayLayer = 0;
    depthStencilViewCI.subresourceRange.layerCount = 1;
    depthStencilViewCI.image = depthStencil.image;
    VK_CHECK(vkCreateImageView(device, &depthStencilViewCI, nullptr, &depthStencil.view));
}

void VulkanBase::createRenderpass() {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = vulkanSwapchain.colorFormat;
    // Related to multisampling
    // If not doing multisampling, set to count 1 bit
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    // LoadOp and storeOp apply to color and depth data
    // Clear the values in the attachment to a constant at the start of rendering
    // Here, it is to clear the framebuffer to black before drawing a new frame
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    // Store rendered contents to be read later
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    // Not using the stencil buffer
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // Layout of pixels that the image will transition to
    // We want it to be ready for presentation using the swap chain after rendering
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = depthFormat;
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };
    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    VK_CHECK(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass));
}

void VulkanBase::createFrameBuffers() {
    swapChainFramebuffers.resize(vulkanSwapchain.imageCount);
    for (size_t i = 0; i < vulkanSwapchain.imageViews.size(); i++) {
        std::array<VkImageView, 2> attachments = {
            vulkanSwapchain.imageViews[i],
            depthStencil.view
        };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = windowWidth;
        framebufferInfo.height = windowHeight;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}

std::vector<char> VulkanBase::readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}


VkShaderModule VulkanBase::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }
    return shaderModule;
}

void VulkanBase::prepare() {
    vulkanSwapchain = vki::VulkanSwapchain(instance, physicalDevice, device, surface);
    vulkanSwapchain.createSwapchain(windowWidth, windowHeight);
    createVmaAllocator();
    createRenderpass();
    prepared = true;
}
void VulkanBase::renderLoop() {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        render();
    }
    vkDeviceWaitIdle(device);
}
void VulkanBase::cleanUp() {
    vkDestroyImageView(device, depthStencil.view, nullptr);
    vmaDestroyImage(allocator, depthStencil.image, depthStencil.allocation);
    glfwDestroyWindow(window);
    glfwTerminate();
    vkDestroyRenderPass(device, renderPass, nullptr);
    for (auto i : swapChainFramebuffers) {
        vkDestroyFramebuffer(device, i, nullptr);
    }
    vkDestroyImageView(device, depthStencil.view, nullptr);
    vmaDestroyImage(allocator, depthStencil.image, depthStencil.allocation);
    vulkanSwapchain.cleanUp();
    vmaDestroyAllocator(allocator);
    vulkanDevice.cleanUp();
}


