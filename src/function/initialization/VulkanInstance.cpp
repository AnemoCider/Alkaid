#include "VulkanInstance.h"

bool vki::Instance::isPhyDeviceSuitable(const vk::PhysicalDevice& device) {
    return true;
}

void vki::Instance::selectDepthFormat() {
    std::vector<vk::Format> formatList = {
        vk::Format::eD32SfloatS8Uint,
        vk::Format::eD32Sfloat,
        vk::Format::eD24UnormS8Uint,
        vk::Format::eD16UnormS8Uint,
        vk::Format::eD16Unorm
    };

    for (vk::Format format : formatList) {
        vk::FormatProperties props;
        props = phyDevice.getFormatProperties(format);

        if (props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment) {
            depthFormat = format;
            break;
        }
    }

    // Log an error if no suitable format is found
    if (depthFormat == vk::Format::eUndefined) {
        throw std::runtime_error("No suitable depth stencil format available.\n");
    }
}

void vki::Instance::init() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    vk::ApplicationInfo appInfo{
        .pApplicationName = "Example",
        .applicationVersion = 1,
        .pEngineName = "Alkaid",
        .engineVersion = 1,
        .apiVersion = VK_API_VERSION_1_3
    };
    
    if (enableValidationLayer) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }

    uint32_t glfwExtCount;
    auto pglfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    for (uint32_t i = 0; i < glfwExtCount; i++) {
        extensions.push_back(pglfwExts[i]);
    }

#ifdef __APPLE__
    extensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif

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
    instance = vk::createInstance(insInfo);

    pickPhysicalDevice();
    selectDepthFormat();
    getGraphicsQueue();

};

void vki::Instance::addExtensions(const std::vector<const char*>& ext) {
    extensions.insert(extensions.end(), ext.begin(), ext.end());
}

void vki::Instance::addLayers(const std::vector<const char*>& layers) {
    this->layers.insert(this->layers.end(), layers.begin(), layers.end());
}

void vki::Instance::clear() {
    instance.destroy();
}

void vki::Instance::setValidationLayer(bool enable) {
    enableValidationLayer = enable;
}

void vki::Instance::pickPhysicalDevice() {
    for (const auto& device : instance.enumeratePhysicalDevices()) {
        if (isPhyDeviceSuitable(device)) {
            phyDevice = device;
            break;
        }
    }
}

void vki::Instance::getGraphicsQueue() {
    int i = 0;
    for (const auto& queueFamily : phyDevice.getQueueFamilyProperties()) {
        if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
            grqFamilyIndex = i;
        }
        i++;
    }
}

void vki::Instance::createWindow(void* ptr) {
    window = glfwCreateWindow(width, height, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, ptr);
    if (!window) {
        throw std::runtime_error("Failed to create glfw window!\n");
    }
    VkSurfaceKHR _surface;
    
    vki::checkResult(glfwCreateWindowSurface(instance, window, nullptr, &_surface));
    surface = vk::SurfaceKHR(_surface);
    getSupports();
}

void vki::Instance::destroyWindow() {
    glfwTerminate();
    instance.destroySurfaceKHR(surface);
    glfwDestroyWindow(window);
}

void vki::Instance::getSupports() {
    supports.capabilities = phyDevice.getSurfaceCapabilitiesKHR(surface);
    supports.formats = phyDevice.getSurfaceFormatsKHR(surface);
    supports.presentModes = phyDevice.getSurfacePresentModesKHR(surface);
    supports.memProperties = phyDevice.getMemoryProperties();
    supports.properties = phyDevice.getProperties();
}



