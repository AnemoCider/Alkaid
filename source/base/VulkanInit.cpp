#include "VulkanInit.h"
#include <array>

VkDeviceQueueCreateInfo vki::init_device_queue_create_info(const uint32_t queueFamilyIndex, const float& queuePriority) {
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

VkWriteDescriptorSet vki::init_write_descriptor_set(const VkDescriptorSet set, const uint32_t binding, const uint32_t arrOffset, const VkDescriptorType descriptorType, const uint32_t descriptorCount, const VkDescriptorBufferInfo* pBufferInfo, const VkDescriptorImageInfo* pImageInfo) {
    VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write.dstSet = set;
    write.dstBinding = binding;
    write.dstArrayElement = arrOffset;
    write.descriptorType = descriptorType;
    write.descriptorCount = descriptorCount;
    write.pBufferInfo = pBufferInfo;
    write.pImageInfo = pImageInfo;
    return write;
}

VkImageCreateInfo vki::init_image_create_info(const VkImageType imageType, const VkFormat format, const VkExtent3D extent, const uint32_t mipCount, const VkSampleCountFlagBits sampleCount, const VkImageTiling tiling, const VkImageUsageFlags usage, const VkImageLayout initLayout, const VkSharingMode shareMode) {
    VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent = extent;
    imageInfo.mipLevels = mipCount;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = sampleCount;
    imageInfo.tiling = tiling;
    imageInfo.usage = usage;
    imageInfo.initialLayout = initLayout;
    imageInfo.sharingMode = shareMode;
    return imageInfo;
}

VkImageMemoryBarrier vki::init_image_memory_barrier(const VkImage& image, const VkImageSubresourceRange& subRange, VkAccessFlags srcAccessMask, VkAccessFlags dstAccesMask, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t srcQueueFamilyIndex, uint32_t dstQueueFamilyIndex) {
    VkImageMemoryBarrier imageMemoryBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    imageMemoryBarrier.srcQueueFamilyIndex = srcQueueFamilyIndex;
    imageMemoryBarrier.dstQueueFamilyIndex = dstQueueFamilyIndex;
    imageMemoryBarrier.image = image;
    imageMemoryBarrier.subresourceRange = subRange;
    imageMemoryBarrier.srcAccessMask = srcAccessMask;
    imageMemoryBarrier.dstAccessMask = dstAccesMask;
    imageMemoryBarrier.oldLayout = oldLayout;
    imageMemoryBarrier.newLayout = newLayout;
    return imageMemoryBarrier;
}

VkImageViewCreateInfo vki::init_image_view_create_info(const VkImageViewType viewType, const VkFormat format, const VkImage& image) {
    VkImageViewCreateInfo view{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    view.viewType = viewType;
    view.format = format;
    view.image = image;
    return view;
}

VkPipelineShaderStageCreateInfo vki::init_pipeline_shader_stage_create_info(const VkShaderStageFlagBits stageBits, const VkShaderModule & module, const char* name) {
    VkPipelineShaderStageCreateInfo shaderStage{};
    shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStage.stage = stageBits;
    shaderStage.module = module;
    shaderStage.pName = name;
    return shaderStage;
}

VkVertexInputBindingDescription vki::init_vertex_input_binding_description(const uint32_t binding, const uint32_t stride, const VkVertexInputRate inputRate) {
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = binding;
    bindingDescription.stride = stride;
    bindingDescription.inputRate = inputRate;
    return bindingDescription;
}

VkVertexInputAttributeDescription vki::init_vertex_input_attribute_description(const uint32_t binding, const uint32_t location, const VkFormat format, const uint32_t offset) {
    VkVertexInputAttributeDescription attributeDescription{};
    attributeDescription.binding = binding;
    attributeDescription.location = location;
    attributeDescription.format = format;
    attributeDescription.offset = offset;
    return attributeDescription;
}

VkPipelineVertexInputStateCreateInfo vki::init_pipeline_vertex_inputState_create_info(const uint32_t vertexBindingDesCount, const uint32_t vertexAttribDesCount, const VkVertexInputBindingDescription* pinputBindingDes, const VkVertexInputAttributeDescription* pInputAttribDes) {
    VkPipelineVertexInputStateCreateInfo vertexInputCreateInfo{};
    vertexInputCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputCreateInfo.vertexBindingDescriptionCount = vertexBindingDesCount;
    vertexInputCreateInfo.vertexAttributeDescriptionCount = vertexAttribDesCount;
    vertexInputCreateInfo.pVertexBindingDescriptions = pinputBindingDes;
    vertexInputCreateInfo.pVertexAttributeDescriptions = pInputAttribDes;
    return vertexInputCreateInfo;
}

VkPipelineInputAssemblyStateCreateInfo vki::init_pipeline_input_assembly_state_create_info(const VkPrimitiveTopology topology, const VkBool32 primitiveRestartEnable) {
    VkPipelineInputAssemblyStateCreateInfo assemblyState{};
    assemblyState.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    assemblyState.topology = topology;
    assemblyState.primitiveRestartEnable = primitiveRestartEnable;
    return assemblyState;
}

VkPipelineViewportStateCreateInfo vki::init_pipeline_viewport_state_create_info(const uint32_t viewportCount, const uint32_t scissorsCount) {
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = viewportCount;
    viewportState.scissorCount = scissorsCount;
    return viewportState;
}

VkPipelineRasterizationStateCreateInfo vki::init_pipeline_rasterization_state_create_info(const VkBool32 depthClampEnable, const VkBool32 rasterizerDiscardEnable, const VkPolygonMode polygonMode, const float lineWidth, const VkCullModeFlags cullMode, const VkFrontFace frontFace, const VkBool32 depthBiasEnable) {
    VkPipelineRasterizationStateCreateInfo rasterizationState{};
    rasterizationState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizationState.depthClampEnable = depthClampEnable;
    rasterizationState.rasterizerDiscardEnable = rasterizerDiscardEnable;
    rasterizationState.polygonMode = polygonMode;
    rasterizationState.lineWidth = lineWidth;
    rasterizationState.cullMode = cullMode;
    rasterizationState.frontFace = frontFace;
    rasterizationState.depthBiasEnable = depthBiasEnable;
    return rasterizationState;
}

VkPipelineDepthStencilStateCreateInfo vki::init_pipeline_depth_stencil_state_create_info(const VkBool32 depthTestEnable, const VkBool32 depthWriteEnable, const VkCompareOp depthCompareOp, const VkBool32 depthBoundsTestEnable, const float minDepthBounds, const float maxDepthBounds, const VkBool32 stencilTestEnable, const VkStencilOpState front, const VkStencilOpState back) {
    VkPipelineDepthStencilStateCreateInfo depthStencilState{};
    depthStencilState.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencilState.depthTestEnable = depthTestEnable;
    depthStencilState.depthWriteEnable = depthWriteEnable;
    depthStencilState.depthCompareOp = depthCompareOp;
    depthStencilState.depthBoundsTestEnable = depthBoundsTestEnable;
    depthStencilState.minDepthBounds = minDepthBounds;
    depthStencilState.maxDepthBounds = maxDepthBounds;
    depthStencilState.stencilTestEnable = stencilTestEnable;
    depthStencilState.front = front;
    depthStencilState.back = back;
    return depthStencilState;
}

VkPipelineMultisampleStateCreateInfo vki::init_pipeline_multisample_state_create_info(const VkBool32 sampleShadingEnable, const VkSampleCountFlagBits rasterizationSamples) {
    VkPipelineMultisampleStateCreateInfo multisampleState{};
    multisampleState.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampleState.sampleShadingEnable = sampleShadingEnable;
    multisampleState.rasterizationSamples = rasterizationSamples;
    return multisampleState;
}

VkPipelineColorBlendAttachmentState vki::init_pipeline_color_blend_attachment_state(const VkColorComponentFlags colorWriteMask, const VkBool32 blendEnable) {
    VkPipelineColorBlendAttachmentState blendAttachmentState{};
    blendAttachmentState.colorWriteMask = colorWriteMask;
    blendAttachmentState.blendEnable = blendEnable;
    return blendAttachmentState;
}

VkPipelineColorBlendStateCreateInfo vki::init_pipeline_color_blend_state_create_info(const VkBool32 logicOpEnable, const VkLogicOp logicOp, const uint32_t attachmentCount, const VkPipelineColorBlendAttachmentState* pAttachments, const std::array<float, 4>& blendConstants) {
    VkPipelineColorBlendStateCreateInfo colorBlendState{};
    colorBlendState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlendState.logicOpEnable = logicOpEnable;
    colorBlendState.logicOp = logicOp;
    colorBlendState.attachmentCount = attachmentCount;
    colorBlendState.pAttachments = pAttachments;
    std::copy(blendConstants.begin(), blendConstants.end(), colorBlendState.blendConstants);
    return colorBlendState;
}

VkPipelineDynamicStateCreateInfo vki::init_pipeline_dynamic_state_create_info(const uint32_t dynamicStateCount, const VkDynamicState* pDynamicStates) {
    VkPipelineDynamicStateCreateInfo dynamicStateCreateInfo{};
    dynamicStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicStateCreateInfo.dynamicStateCount = dynamicStateCount;
    dynamicStateCreateInfo.pDynamicStates = pDynamicStates;
    return dynamicStateCreateInfo;
}

VkPipelineLayoutCreateInfo vki::init_pipeline_layout_create_info(const uint32_t setLayoutCount, const VkDescriptorSetLayout* pSetLayouts, const uint32_t pushConstantRangeCount) {
    VkPipelineLayoutCreateInfo layoutCreateInfo{};
    layoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCreateInfo.setLayoutCount = setLayoutCount;
    layoutCreateInfo.pSetLayouts = pSetLayouts;
    layoutCreateInfo.pushConstantRangeCount = pushConstantRangeCount;
    return layoutCreateInfo;
}
