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

		struct Setting {
			vk::SurfaceFormatKHR surfaceFormat;
			vk::PresentModeKHR presentMode;
			vk::Extent2D extent;
			uint32_t imageCount;
		} setting;
		
		void setUp();
		void createViews();
	public:
		std::vector<vk::ImageView> views;
		void setInstance(vki::Instance* inst);
		void setDevice(vki::Device* device);
		void init();
		void clear();
		uint32_t getImageCount() const;
		vk::Format getColorFormat() const;
		vk::SwapchainKHR& getSwapChain();
	};
}