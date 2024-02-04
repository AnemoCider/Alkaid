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
		std::vector<vk::ImageView> views;

		struct Setting {
			vk::SurfaceFormatKHR surfaceFormat;
			vk::PresentModeKHR presentMode;
			vk::Extent2D extent;
			uint32_t imageCount;
		} setting;

		
		void setUp();
	public:
		void setInstance(vki::Instance* inst);
		void setDevice(vki::Device* device);
		void init();
		void clear();
		void createViews();
	};
}