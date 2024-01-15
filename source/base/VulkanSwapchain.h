#pragma once

#include <vector>

#include <VulkanCommon.h>

namespace vki {
class VulkanSwapchain {

private:
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

public:
    VulkanSwapchain() = default;
    VulkanSwapchain(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device, VkSurfaceKHR surface);

    VkFormat colorFormat = VK_FORMAT_B8G8R8A8_SRGB;
	VkColorSpaceKHR colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
	uint32_t imageCount;
	std::vector<VkImage> images;
	std::vector<VkImageView> imageViews;
	uint32_t queueNodeIndex = UINT32_MAX;    

    VkSurfaceKHR surface;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    void createSwapchain(uint32_t& windowWidth, uint32_t& windowHeight);
    void cleanUp();
};
};