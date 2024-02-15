#pragma once

#include "common/VulkanCommon.h"

namespace vki {
    class Buffer {
    public:
        vk::Buffer buffer;
        vk::DeviceMemory mem;
    };

    class StagingBuffer : public Buffer{
    public:
        /**
         * @brief create a bufferCreateInfo for staging buffer
         * @param sz size of the buffer
         * @return created buffer create info
        */
        static vk::BufferCreateInfo getCI(vk::DeviceSize sz);
    };

    class UniformBuffer : public Buffer{
    public:
		void* mapped{ nullptr };
	};
};