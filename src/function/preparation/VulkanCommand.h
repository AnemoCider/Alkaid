#pragma once

#include "common/VulkanCommon.h"
#include "initialization/VulkanDevice.h"

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