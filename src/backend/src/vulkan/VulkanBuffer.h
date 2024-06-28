#pragma once

#include "common/VulkanCommon.h"
#include "initialization/VulkanDevice.h"
#include <vector>

namespace vki {

    class Buffer {
    public:
        vk::Buffer buffer;
        vk::DeviceMemory mem;
        void clear(vki::Device& device);
        Buffer() = default;
        Buffer(vki::Device& device, vk::DeviceSize sz, const vk::Flags<vk::BufferUsageFlagBits> usage, const vk::Flags<vk::MemoryPropertyFlagBits>& memoryProps);
    };

    class StagingBuffer : public Buffer{
    public:
        /**
         * @brief create a bufferCreateInfo for staging buffer
         * @param sz size of the buffer
         * @return created buffer create info
        */
        StagingBuffer() = default;
        StagingBuffer(vki::Device& device, vk::DeviceSize sz);
    };

    class UniformBuffer : public Buffer{
    public:
        UniformBuffer() = default;
        UniformBuffer(vki::Device& device, vk::DeviceSize sz);
		void* mapped{ nullptr };
	};
};


