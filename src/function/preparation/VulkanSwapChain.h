#pragma once

#include "common/VulkanCommon.h"
#include "initialization/VulkanInstance.h"
#include "initialization/VulkanDevice.h"

namespace vki {
	class SwapChain {
	private:
		vki::Instance* instance;
		vki::Device* device;
		vk::SwapchainKHR swapChain{ nullptr };

		struct SupportDetails {
			vk::SurfaceCapabilitiesKHR capabilities;
			std::vector<vk::SurfaceFormatKHR> formats;
			std::vector<vk::PresentModeKHR> presentModes;
		} supports;
		struct Setting {
			vk::SurfaceFormatKHR surfaceFormat;
			vk::PresentModeKHR presentMode;
			vk::Extent2D extent;
			uint32_t imageCount;
		} setting;
		void getSurfaceSupports();
		void setUp();
	public:
		void setInstance(vki::Instance* inst);
		void setDevice(vki::Device* device);
		void init();
		void clear();
	};
}