#include "utils/VulkanBuffer.h"
#include "VulkanBuffer.h"

vk::BufferCreateInfo vki::StagingBuffer::getCI(vk::DeviceSize sz)
{
    vk::BufferCreateInfo ci {
        .size = sz,
        .usage = vk::BufferUsageFlagBits::eTransferSrc
    };
    return ci;
}