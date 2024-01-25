#pragma once

#include "VulkanCommon.h"

namespace vki {
    VkDeviceQueueCreateInfo init_device_queue_create_info(
        const uint32_t queueFamilyIndex, 
        const float& queuePriority);
    VkCommandPoolCreateInfo init_command_pool_create_info(
        const uint32_t queueFamilyIndex, 
        const VkCommandPoolCreateFlagBits flags);
    VkCommandBufferAllocateInfo init_command_buffer_allocate_info(
        const uint32_t bufferCount, 
        const VkCommandPool cmdPool,
        const VkCommandBufferLevel bufferLevel = VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    VkDescriptorPoolCreateInfo init_descriptor_pool_create_info(
        const uint32_t sizeCount, 
        const VkDescriptorPoolSize* pSizes, 
        const uint32_t maxSetsCount);
    VkDescriptorSetAllocateInfo init_descriptor_set_allocate_info(
        const VkDescriptorPool descriptorPool, 
        const uint32_t setCount, 
        const VkDescriptorSetLayout* pLayouts);
    VkWriteDescriptorSet init_write_descriptor_set(
        const VkDescriptorSet set, 
        const uint32_t binding, 
        const uint32_t arrOffset, 
        const VkDescriptorType descriptorType, 
        const uint32_t descriptorCount, 
        const VkDescriptorBufferInfo* pBufferInfo, 
        const VkDescriptorImageInfo* pImageInfo);
	VkImageCreateInfo init_image_create_info(
        const VkImageType imageType, 
        const VkFormat format, 
        const VkExtent3D extent, 
        const uint32_t mipCount, 
        const VkSampleCountFlagBits sampleCount, 
        const VkImageTiling tiling, 
        const VkImageUsageFlags usage, 
        const VkImageLayout initLayout = VK_IMAGE_LAYOUT_UNDEFINED, 
        const VkSharingMode shareMode = VK_SHARING_MODE_EXCLUSIVE);
    VkImageMemoryBarrier init_image_memory_barrier(
        const VkImage& image, 
        const VkImageSubresourceRange& subRange, 
        VkAccessFlags srcAccessMask, 
        VkAccessFlags dstAccesMask, 
        VkImageLayout oldLayout, 
        VkImageLayout newLayout, 
        uint32_t srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, 
        uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED);
    VkImageViewCreateInfo init_image_view_create_info(
        const VkImageViewType viewType, 
        const VkFormat format, 
        const VkImage& image);

    // Pipeline Stuff
    VkPipelineShaderStageCreateInfo init_pipeline_shader_stage(
        const VkShaderStageFlagBits stageBits, 
        const VkShaderModule& module, 
        const char* name);

    VkVertexInputBindingDescription init_vertex_input_binding_description(
        const uint32_t binding, 
        const uint32_t stride, 
        const VkVertexInputRate inputRate = VK_VERTEX_INPUT_RATE_VERTEX);

    VkVertexInputAttributeDescription init_vertex_input_attribute_description(
        const uint32_t binding, 
        const uint32_t location, 
        const VkFormat format, 
        const uint32_t offset);


    VkPipelineVertexInputStateCreateInfo init_pipeline_vertex_inputState_create_info(
        const uint32_t vertexBindingDesCount, 
        const uint32_t vertexAttribDesCount, 
        const VkVertexInputBindingDescription* pinputBindingDes, 
        const VkVertexInputAttributeDescription* pInputAttribDes);

    VkPipelineInputAssemblyStateCreateInfo init_input_assembly_state_create_info(
        const VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 
        const VkBool32 primitiveRestartEnable = false);

    VkPipelineViewportStateCreateInfo init_pipeline_viewport_state_create_info(
        const uint32_t viewportCount = 1, 
        const uint32_t scissorsCount = 1);


    VkPipelineRasterizationStateCreateInfo init_pipeline_rasterization_state_create_info(
        const VkBool32 depthClampEnable, 
        const VkBool32 rasterizerDiscardEnable, 
        const VkPolygonMode polygonMode, 
        const float lineWidth,
        const VkCullModeFlags cullMode, 
        const VkFrontFace frontFace, 
        const VkBool32 depthBiasEnable);
      

    VkPipelineDepthStencilStateCreateInfo init_pipeline_depth_stencil_state_create_info(
        const VkBool32 depthTestEnable,
        const VkBool32 depthWriteEnable,
        const VkCompareOp depthCompareOp,
        const VkBool32 depthBoundsTestEnable,
        const float minDepthBounds,
        const float maxDepthBounds,
        const VkBool32 stencilTestEnable,
        const VkStencilOpState front,
        const VkStencilOpState back);

    VkPipelineMultisampleStateCreateInfo init_pipeline_multisample_state_create_info(
        const VkBool32 sampleShadingEnable,
        const VkSampleCountFlagBits rasterizationSamples);

    VkPipelineColorBlendAttachmentState init_pipeline_color_blend_attachment_state(
        const VkColorComponentFlags colorWriteMask,
        const VkBool32 blendEnable);

    VkPipelineColorBlendStateCreateInfo init_pipeline_color_blend_state_create_info(
        const VkBool32 logicOpEnable,
        const VkLogicOp logicOp,
        const uint32_t attachmentCount,
        const VkPipelineColorBlendAttachmentState* pAttachments,
        const std::array<float, 4>& blendConstants);

    VkPipelineDynamicStateCreateInfo init_pipeline_dynamic_state_create_info(
        const uint32_t dynamicStateCount,
        const VkDynamicState* pDynamicStates);


    VkPipelineLayoutCreateInfo init_pipeline_layout_create_info(
        const uint32_t setLayoutCount,
        const VkDescriptorSetLayout* pSetLayouts,
        const uint32_t pushConstantRangeCount);
};



