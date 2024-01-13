#include <vulkan/vk_mem_alloc.h>
#include "VulkanBase.h"

#include <memory>
#include <iostream>

class VulkanTriangle : public VulkanBase {

};

int main() {
    auto app = std::make_unique<VulkanTriangle>();
    app->createWindow();
    app->initVulkan();
    app->prepare();
    app->renderLoop();
    std::cout << "Completed!!\n";
}