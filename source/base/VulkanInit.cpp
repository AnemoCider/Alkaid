#include "VulkanInit.h"


VkDeviceQueueCreateInfo vki::init_device_queue_create_info(const uint32_t queueFamilyIndex) {
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    return queueCreateInfo;
}

VkCommandPoolCreateInfo vki::init_command_pool_create_info(const uint32_t queueFamilyIndex, const VkCommandPoolCreateFlagBits flags) {
    VkCommandPoolCreateInfo poolCreateInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    poolCreateInfo.queueFamilyIndex = queueFamilyIndex;
    poolCreateInfo.flags = flags;
    return poolCreateInfo;
}

VkCommandBufferAllocateInfo vki::init_command_buffer_allocate_info(const uint32_t bufferCount, const VkCommandPool cmdPool, VkCommandBufferLevel bufferLevel) {
    VkCommandBufferAllocateInfo bufferAllocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    bufferAllocInfo.commandBufferCount = bufferCount;
    bufferAllocInfo.commandPool = cmdPool;
    bufferAllocInfo.level = bufferLevel;
    return bufferAllocInfo;
}

VkDescriptorPoolCreateInfo vki::init_descriptor_pool_create_info(const uint32_t sizeCount, const VkDescriptorPoolSize* pSizes, const uint32_t maxSetsCount) {
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.poolSizeCount = sizeCount;
    poolInfo.pPoolSizes = pSizes;
    poolInfo.maxSets = maxSetsCount;
    return poolInfo;
}

VkDescriptorSetAllocateInfo vki::init_descriptor_set_allocate_info(const VkDescriptorPool descriptorPool, const uint32_t setCount, const VkDescriptorSetLayout* pLayouts) {
    VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    setAllocInfo.descriptorPool = descriptorPool;
    setAllocInfo.descriptorSetCount = setCount;
    setAllocInfo.pSetLayouts = pLayouts;
    return setAllocInfo;
}

VkWriteDescriptorSet vki::init_write_descriptor_set(const VkDescriptorSet set, const uint32_t binding, const uint32_t arrOffset, const VkDescriptorType descriptorType, const uint32_t descriptorCount, const VkDescriptorBufferInfo* pBufferInfo) {
    VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write.dstSet = set;
    write.dstBinding = binding;
    write.dstArrayElement = arrOffset;
    write.descriptorType = descriptorType;
    write.descriptorCount = descriptorCount;
    write.pBufferInfo = pBufferInfo;
    return write;
}
