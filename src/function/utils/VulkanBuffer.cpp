#include "utils/VulkanBuffer.h"

vki::StagingBuffer::StagingBuffer(vki::Device& device, vk::DeviceSize sz) : 
	Buffer(device, sz, vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent) {}

vki::Buffer::Buffer(vki::Device& device, vk::DeviceSize sz, const vk::Flags<vk::BufferUsageFlagBits> usage, const vk::Flags<vk::MemoryPropertyFlagBits>& memoryProps) {
	vk::BufferCreateInfo bufferCI{
		.size = sz,
		.usage = usage
	};

	buffer = device.getDevice().createBuffer(bufferCI);
	auto memReq = device.getDevice().getBufferMemoryRequirements(buffer);
	vk::MemoryAllocateInfo memAI{
		.allocationSize = memReq.size,
		.memoryTypeIndex = device.getMemoryType(memReq.memoryTypeBits, memoryProps)
	};
	mem = device.getDevice().allocateMemory(memAI);
	device.getDevice().bindBufferMemory(buffer, mem, sz);
}

void vki::Buffer::clear(vki::Device& device) {
	device.getDevice().freeMemory(mem);
	device.getDevice().destroyBuffer(buffer);
}

vki::UniformBuffer::UniformBuffer(vki::Device& device, vk::DeviceSize sz) :
	Buffer(device, sz, vk::BufferUsageFlagBits::eUniformBuffer,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent) {}
	