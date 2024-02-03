#include "VulkanInstance.h"

bool vki::Instance::isPhyDeviceSuitable(const vk::PhysicalDevice& device) {
    return true;
}

void vki::Instance::init() {
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

    vk::InstanceCreateInfo insInfo{
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = static_cast<uint32_t>(layers.size()),
        .ppEnabledLayerNames = layers.data(),
        .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data()
    };

#ifdef __APPLE__
    insInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    extensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif

    instance = vk::createInstance(insInfo);

    pickPhysicalDevice();
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
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(width, height, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, ptr);
    if (!window) {
        throw std::runtime_error("Failed to create glfw window!\n");
    }
    glfwCreateWindowSurface(instance, window, nullptr, reinterpret_cast<VkSurfaceKHR*>(&surface));
}

void vki::Instance::destroyWindow() {
    glfwTerminate();
    instance.destroySurfaceKHR(surface);
    glfwDestroyWindow(window);
}



