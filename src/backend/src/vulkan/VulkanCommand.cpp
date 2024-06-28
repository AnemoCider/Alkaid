#include "preparation/VulkanCommand.h"

void vki::Command::setDevice(vki::Device* device) {
	this->device = device;
}

void vki::Command::init() {
	vk::CommandPoolCreateInfo poolCI{
		.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
		.queueFamilyIndex = device->getGrqFamilyIndex(),
	};
	commandPool = device->getDevice().createCommandPool(poolCI);
}

void vki::Command::clear() {
	device->getDevice().destroyCommandPool(commandPool);
}

vk::CommandBuffer vki::Command::createBuffer() {
	vk::CommandBufferAllocateInfo bufferAI{
		.commandPool = commandPool,
		.level = vk::CommandBufferLevel::ePrimary,
		.commandBufferCount = 1
	};

	return device->getDevice().allocateCommandBuffers(bufferAI).front();
}