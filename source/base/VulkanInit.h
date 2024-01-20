#pragma once

#include "VulkanCommon.h"

namespace vki {
    VkDeviceQueueCreateInfo init_device_queue_create_info(const uint32_t queueFamilyIndex, const float& queuePriority);
    VkCommandPoolCreateInfo init_command_pool_create_info(const uint32_t queueFamilyIndex, const VkCommandPoolCreateFlagBits flags);
    VkCommandBufferAllocateInfo init_command_buffer_allocate_info(const uint32_t bufferCount, const VkCommandPool cmdPool, const VkCommandBufferLevel bufferLevel = VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    VkDescriptorPoolCreateInfo init_descriptor_pool_create_info(const uint32_t sizeCount, const VkDescriptorPoolSize* pSizes, const uint32_t maxSetsCount);
    VkDescriptorSetAllocateInfo init_descriptor_set_allocate_info(const VkDescriptorPool descriptorPool, const uint32_t setCount, const VkDescriptorSetLayout* pLayouts);
    VkWriteDescriptorSet init_write_descriptor_set(const VkDescriptorSet set, const uint32_t binding, const uint32_t arrOffset, const VkDescriptorType descriptorType, const uint32_t descriptorCount, const VkDescriptorBufferInfo* pBufferInfo, const VkDescriptorImageInfo* pImageInfo);
	VkImageCreateInfo init_image_create_info(const VkImageType imageType, const VkFormat format, const VkExtent3D extent, const uint32_t mipCount, const VkSampleCountFlagBits sampleCount, const VkImageTiling tiling, const VkImageUsageFlags usage, const VkImageLayout initLayout = VK_IMAGE_LAYOUT_UNDEFINED, const VkSharingMode shareMode = VK_SHARING_MODE_EXCLUSIVE);
};

