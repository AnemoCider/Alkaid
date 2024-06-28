#pragma once

#include "VulkanCommon.h"
#include "VulkanDevice.h"

namespace vki {
	class Command {
	private:
	public:
		vk::CommandPool commandPool{ nullptr };
		vki::Device* device;

		void setDevice(vki::Device* device);
		void init();
		void clear();

		vk::CommandBuffer createBuffer();
	};
};