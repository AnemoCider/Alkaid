#include <vulkan/vk_mem_alloc.h>
#include "VulkanBase.h"

#include <memory>
#include <iostream>
#include <array>

class VulkanTriangle : public VulkanBase {

private:
    static constexpr uint32_t maxFrameCount = 2;
    uint32_t currentFrame = 0;
    std::array<VkSemaphore, maxFrameCount> presentCompleteSemaphores {};
    std::array<VkSemaphore, maxFrameCount> renderCompleteSemaphores {};
    std::array<VkFence, maxFrameCount> waitFences {};

    void createSyncObjects() {
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < maxFrameCount; i++) {
            VK_CHECK(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &presentCompleteSemaphores[i]));
            VK_CHECK(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderCompleteSemaphores[i]));
            VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &waitFences[i]));
        }
    }

    void createCommandBuffers() {
        VkCommandPoolCreateInfo poolCreateInfo {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolCreateInfo.queueFamilyIndex = vulkanDevice.queueFamilyIndices.graphics.value();
        poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CHECK(vkCreateCommandPool(device, &poolCreateInfo, nullptr, &commandPool));

        commandBuffers.resize(maxFrameCount);
        VkCommandBufferAllocateInfo bufferAllocInfo {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        bufferAllocInfo.commandBufferCount = commandBuffers.size();
        bufferAllocInfo.commandPool = commandPool;
        bufferAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        VK_CHECK(vkAllocateCommandBuffers(device, &bufferAllocInfo, commandBuffers.data()));
    }

    std::string getShaderPathName() {
        return "../../shaders/basicTriangle/basicTriangle";
    }

    void recreateSwapChain() {
        prepared = false;
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device);

        for (auto i : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, i, nullptr);
        }
        vkDestroyImageView(device, depthStencil.view, nullptr);
        vmaDestroyImage(allocator, depthStencil.image, depthStencil.allocation);
        vulkanSwapchain.cleanUp();

        windowWidth = (uint32_t)width;
        windowHeight = (uint32_t)height;

        vulkanSwapchain.createSwapchain(windowWidth, windowHeight);
        createDepthStencil();
        createFrameBuffers();
        prepared = true;
    }

public:
    void prepare() {
		VulkanBase::prepare();
		createSyncObjects();
		createCommandBuffers();
		// createVertexBuffer();
		// createUniformBuffers();
		// createDescriptorSetLayout();
		// createDescriptorPool();
		// createDescriptorSets();
		// createPipelines();
		prepared = true;
	}

    void cleanUp() {
        for (size_t i = 0; i < maxFrameCount; i++) {
            vkDestroySemaphore(device, presentCompleteSemaphores[i], nullptr);
            vkDestroySemaphore(device, renderCompleteSemaphores[i], nullptr);
            vkDestroyFence(device, waitFences[i], nullptr);
        }
        VulkanBase::cleanUp();
        
    }

    void render() {
        if (!prepared)
            return;
        vkWaitForFences(device, 1, &waitFences[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, vulkanSwapchain.swapchain, UINT64_MAX, presentCompleteSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
        
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return;
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        // Set fence to unsignaled
        // Must delay this to after recreateSwapChain to avoid deadlock
        vkResetFences(device, 1, &waitFences[currentFrame]);

        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        // recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        // updateUniformBuffer(currentFrame);

        // VkSubmitInfo submitInfo{};
        // submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        // // Which semaphores to wait
        // VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
        // // On which stage to wait
        // // Here we want to wait with writing colors to the image until it's available
        // VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        // submitInfo.waitSemaphoreCount = 1;
        // submitInfo.pWaitSemaphores = waitSemaphores;
        // submitInfo.pWaitDstStageMask = waitStages;

        // submitInfo.commandBufferCount = 1;
        // submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

        // // Which semaphores to signal once the command buffer(s) has finished execution
        // VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
        // submitInfo.signalSemaphoreCount = 1;
        // submitInfo.pSignalSemaphores = signalSemaphores;

        // if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
        //     throw std::runtime_error("failed to submit draw command buffer!");
        // }

        // VkPresentInfoKHR presentInfo{};
        // presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        // presentInfo.waitSemaphoreCount = 1;
        // presentInfo.pWaitSemaphores = signalSemaphores;

        // VkSwapchainKHR swapChains[] = { swapChain };
        // presentInfo.swapchainCount = 1;
        // presentInfo.pSwapchains = swapChains;
        // presentInfo.pImageIndices = &imageIndex;

        // presentInfo.pResults = nullptr; // Optional

        // result = vkQueuePresentKHR(vulkanDevice.queueFamilyIndices.graphics, &presentInfo);
        // if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
        //     framebufferResized = false;
        //     recreateSwapChain();
        // } else if (result != VK_SUCCESS) {
        //     throw std::runtime_error("failed to present swap chain image!");
        // }
        currentFrame = (currentFrame + 1) % maxFrameCount;
    }
};

int main() {
    auto app = std::make_unique<VulkanTriangle>();
    app->createWindow();
    app->initVulkan();
    app->prepare();
    app->renderLoop();
    app->cleanUp();
    std::cout << "Completed!!\n";
}