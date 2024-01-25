#include <vulkan/vk_mem_alloc.h>
#include "VulkanBase.h"

#include <ktx.h>
#include <ktxvulkan.h>

#include <memory>
#include <iostream>
#include <array>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <chrono>

#include "VulkanInit.h"

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;
};

struct Staging {
    VkBuffer buffer;
    VmaAllocation alloc;
};

const std::vector<Vertex> verticesData = {
    {{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},

    {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
    {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}}
};

const std::vector<uint16_t> indicesData = {
    0, 1, 2, 2, 3, 0,
    4, 5, 6, 6, 7, 4
};

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

class VulkanTriangle : public VulkanBase {

private:
    bool framebufferResized = false;

    static constexpr uint32_t maxFrameCount = 2;
    uint32_t currentFrame = 0;
    std::array<VkSemaphore, maxFrameCount> presentCompleteSemaphores {};
    std::array<VkSemaphore, maxFrameCount> renderCompleteSemaphores {};
    std::array<VkFence, maxFrameCount> waitFences {};
    

    VkDescriptorPool descriptorPool;
    VkDescriptorSetLayout descriptorSetLayout;
    std::array<VkDescriptorSet, maxFrameCount> descriptorSets;
    
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    struct {
		VmaAllocation alloc;
		VkBuffer buffer;
	} vertices;

    struct {
		VmaAllocation alloc;
		VkBuffer buffer;
		uint32_t count{ 0 };
	} indices;

    struct UniformBuffer {
		VmaAllocation alloc;
		VkBuffer buffer;
		void* mapped{ nullptr };
	};

    std::array<UniformBuffer, maxFrameCount> uniformBuffers;

    struct ShaderData {
		glm::mat4 projectionMatrix;
		glm::mat4 modelMatrix;
		glm::mat4 viewMatrix;
	};

    struct Texture {
        VkSampler sampler;
        VkImage image;
        VkImageLayout imageLayout;
        VmaAllocation alloc;
        VkImageView view;
        uint32_t width, height;
        uint32_t mipLevels;
    } texture;

    std::string getShaderPathName() {
        return "shaders/basicTriangle/basicTriangle";
    }
    std::string getAssetPath() {
        return "Vulkan-assets/";
    }

    void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<VulkanTriangle*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void createSyncObjects() {
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < maxFrameCount; i++) {
            VK_CHECK(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &presentCompleteSemaphores[i]));
            VK_CHECK(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderCompleteSemaphores[i]));
            VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &waitFences[i]));
        }
    }

    void createCommandBuffers() {
        auto poolCreateInfo = vki::init_command_pool_create_info(vulkanDevice.queueFamilyIndices.graphics.value(),
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
        VK_CHECK(vkCreateCommandPool(device, &poolCreateInfo, nullptr, &commandPool));

        commandBuffers.resize(maxFrameCount);
        auto bufferAllocInfo = vki::init_command_buffer_allocate_info(commandBuffers.size(), commandPool);
        VK_CHECK(vkAllocateCommandBuffers(device, &bufferAllocInfo, commandBuffers.data()));
    }
    
    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding uboLayoutBinding {};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutBinding textureLayoutBinding{};
        textureLayoutBinding.binding = 1;
        textureLayoutBinding.descriptorCount = 1;
        textureLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        textureLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings{ uboLayoutBinding , textureLayoutBinding };
        VkDescriptorSetLayoutCreateInfo layoutInfo { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutInfo.bindingCount = setLayoutBindings.size();
        layoutInfo.pBindings = setLayoutBindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout));
    }

    void createDescriptorPool() {
        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = 1;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[1].descriptorCount = 1;

        auto poolInfo = vki::init_descriptor_pool_create_info(poolSizes.size(), poolSizes.data(), maxFrameCount);
        vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
    }

    void createDescriptorSets() {
        std::vector<VkDescriptorSetLayout> setLayouts(maxFrameCount, descriptorSetLayout);

        auto setAllocInfo = vki::init_descriptor_set_allocate_info(descriptorPool, descriptorSets.size(), setLayouts.data());
        vkAllocateDescriptorSets(device, &setAllocInfo, descriptorSets.data());

        for (size_t i = 0; i < maxFrameCount; i++) {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i].buffer;
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = texture.imageLayout;
            imageInfo.imageView = texture.view;
            imageInfo.sampler = texture.sampler;

            std::array<VkWriteDescriptorSet, 2> descriptorWrites{};
            descriptorWrites[0] = vki::init_write_descriptor_set(descriptorSets[i], 0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, &bufferInfo, nullptr);
            descriptorWrites[1] = vki::init_write_descriptor_set(descriptorSets[i], 1, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, nullptr, &imageInfo);

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }

    void createVertexBuffer() {
        VkDeviceSize bufferSize = sizeof(verticesData[0]) * verticesData.size();

        Staging stagingBuffer{};

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
    VMA_ALLOCATION_CREATE_MAPPED_BIT, stagingBuffer.buffer, stagingBuffer.alloc, nullptr);

        void* data;

        vmaMapMemory(allocator, stagingBuffer.alloc, &data);
        memcpy(data, verticesData.data(), (size_t)bufferSize);
        vmaUnmapMemory(allocator, stagingBuffer.alloc);

        createBuffer(bufferSize, 
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 
            0, vertices.buffer, vertices.alloc);
        copyBuffer(stagingBuffer.buffer, vertices.buffer, bufferSize);

        vmaDestroyBuffer(allocator, stagingBuffer.buffer, stagingBuffer.alloc);
    }

    void createIndexBuffer() {
        VkDeviceSize bufferSize = sizeof(indicesData[0]) * indicesData.size();

        Staging stagingBuffer{};

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
    VMA_ALLOCATION_CREATE_MAPPED_BIT, stagingBuffer.buffer, stagingBuffer.alloc);

        void* data;

        vmaMapMemory(allocator, stagingBuffer.alloc, &data);
        memcpy(data, indicesData.data(), (size_t)bufferSize);
        vmaUnmapMemory(allocator, stagingBuffer.alloc);

        createBuffer(bufferSize, 
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, 
            0, indices.buffer, indices.alloc);
        copyBuffer(stagingBuffer.buffer, indices.buffer, bufferSize);

        vmaDestroyBuffer(allocator, stagingBuffer.buffer, stagingBuffer.alloc);
    }

    void createUniformBuffers() {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        for (size_t i = 0; i < maxFrameCount; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, 
            uniformBuffers[i].buffer, uniformBuffers[i].alloc);
            vmaMapMemory(allocator, uniformBuffers[i].alloc, &uniformBuffers[i].mapped);
        }
    }

    void createTextureImage() {
        std::string filename = getAssetPath() + "textures/metalplate01_rgba.ktx";
        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

        ktxResult result;
        ktxTexture* ktxTexture;
        result = ktxTexture_CreateFromNamedFile(filename.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktxTexture);
        assert(result == KTX_SUCCESS);
        texture.width = ktxTexture->baseWidth;
        texture.height = ktxTexture->baseHeight;
        texture.mipLevels = ktxTexture->numLevels;
        ktx_uint8_t* ktxTextureData = ktxTexture_GetData(ktxTexture);
        ktx_size_t ktxTextureSize = ktxTexture_GetSize(ktxTexture);

        Staging stagingBuffer{};

        createBuffer(ktxTextureSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT, stagingBuffer.buffer, stagingBuffer.alloc);

        // Copy texture data into host local staging buffer
        void* data;
        VK_CHECK(vmaMapMemory(allocator, stagingBuffer.alloc, &data));
        memcpy(data, ktxTextureData, ktxTextureSize);
        vmaUnmapMemory(allocator, stagingBuffer.alloc);

        // Setup buffer copy regions for each mip level
        std::vector<VkBufferImageCopy> bufferCopyRegions;

        for (uint32_t i = 0; i < texture.mipLevels; i++) {
            // Calculate offset into staging buffer for the current mip level
            ktx_size_t offset;
            KTX_error_code ret = ktxTexture_GetImageOffset(ktxTexture, i, 0, 0, &offset);
            assert(ret == KTX_SUCCESS);
            // Setup a buffer image copy structure for the current mip level
            VkBufferImageCopy bufferCopyRegion = {};
            bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            bufferCopyRegion.imageSubresource.mipLevel = i;
            bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
            bufferCopyRegion.imageSubresource.layerCount = 1;
            bufferCopyRegion.imageExtent.width = ktxTexture->baseWidth >> i;
            bufferCopyRegion.imageExtent.height = ktxTexture->baseHeight >> i;
            bufferCopyRegion.imageExtent.depth = 1;
            bufferCopyRegion.bufferOffset = offset;
            bufferCopyRegions.push_back(bufferCopyRegion);
        }

        // Create optimal tiled target image on the device
        VkImageCreateInfo imageCreateInfo = vki::init_image_create_info(
            VK_IMAGE_TYPE_2D, format, { texture.width, texture.height, 1 }, texture.mipLevels,
            VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

        createImage(imageCreateInfo, texture.image, texture.alloc);

        // Image memory barriers for the texture image

        // The sub resource range describes the regions of the image that will be transitioned using the memory barriers below
        VkImageSubresourceRange subresourceRange = {};
        // Image only contains color data
        subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        // Start at first mip level
        subresourceRange.baseMipLevel = 0;
        // We will transition on all mip levels
        subresourceRange.levelCount = texture.mipLevels;
        // The 2D texture only has one layer
        subresourceRange.layerCount = 1;

        // Transition the texture image layout to transfer target, so we can safely copy our buffer data to it.
        VkImageMemoryBarrier imageMemoryBarrier = vki::init_image_memory_barrier(texture.image, subresourceRange, 0, VK_ACCESS_TRANSFER_WRITE_BIT, 
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        VkCommandBuffer commandBuffer = beginSingleTimeCommands();
        // Insert a memory dependency at the proper pipeline stages that will execute the image layout transition
        // Source pipeline stage is host write/read execution (VK_PIPELINE_STAGE_HOST_BIT)
        // Destination pipeline stage is copy command execution (VK_PIPELINE_STAGE_TRANSFER_BIT)
        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &imageMemoryBarrier);

        // Copy mip levels from staging buffer
        vkCmdCopyBufferToImage(
            commandBuffer,
            stagingBuffer.buffer,
            texture.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            static_cast<uint32_t>(bufferCopyRegions.size()),
            bufferCopyRegions.data());

        
        // Once the data has been uploaded we transfer to the texture image to the shader read layout, so it can be sampled from
        imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Insert a memory dependency at the proper pipeline stages that will execute the image layout transition
        // Source pipeline stage is copy command execution (VK_PIPELINE_STAGE_TRANSFER_BIT)
        // Destination pipeline stage fragment shader access (VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &imageMemoryBarrier);

        endSingleTimeCommands(commandBuffer);
        // Store current layout for later reuse
        texture.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Clean up staging resources
        vmaDestroyBuffer(allocator, stagingBuffer.buffer, stagingBuffer.alloc);

        ktxTexture_Destroy(ktxTexture);

        // Create a texture sampler
        // In Vulkan textures are accessed by samplers
        // This separates all the sampling information from the texture data. This means you could have multiple sampler objects for the same texture with different settings
        // Note: Similar to the samplers available with OpenGL 3.3
        VkSamplerCreateInfo sampler{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        sampler.magFilter = VK_FILTER_LINEAR;
        sampler.minFilter = VK_FILTER_LINEAR;
        sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler.mipLodBias = 0.0f;
        sampler.compareOp = VK_COMPARE_OP_NEVER;
        sampler.minLod = 0.0f;
        // Set max level-of-detail to mip level count of the texture
        sampler.maxLod = (float)texture.mipLevels;
        // Enable anisotropic filtering
        // This feature is optional, so we must check if it's supported on the device
        // TODO: Check it
        // Use max. level of anisotropy for this example
        sampler.maxAnisotropy = physicalDeviceProperties.limits.maxSamplerAnisotropy;
        sampler.anisotropyEnable = VK_TRUE;

        sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        VK_CHECK(vkCreateSampler(device, &sampler, nullptr, &texture.sampler));

        // Create image view
        // Textures are not directly accessed by the shaders and
        // are abstracted by image views containing additional
        // information and sub resource ranges
        VkImageViewCreateInfo view = vki::init_image_view_create_info(VK_IMAGE_VIEW_TYPE_2D, format, texture.image);
        // The subresource range describes the set of mip levels (and array layers) that can be accessed through this image view
        // It's possible to create multiple image views for a single image referring to different (and/or overlapping) ranges of the image
        view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view.subresourceRange.baseMipLevel = 0;
        view.subresourceRange.baseArrayLayer = 0;
        view.subresourceRange.layerCount = 1;
        // Linear tiling usually won't support mip maps
        // Only set mip map count if optimal tiling is used
        view.subresourceRange.levelCount = texture.mipLevels;
        VK_CHECK(vkCreateImageView(device, &view, nullptr, &texture.view));
    }

    void createPipeline() {
        VkShaderModule vertShaderModule = createShaderModule(readFile(getShaderPathName() + ".vert.spv"));
        VkShaderModule fragShaderModule = createShaderModule(readFile(getShaderPathName() + ".frag.spv"));
        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";
        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";
        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        VkVertexInputBindingDescription vertexInputBinding {};
        vertexInputBinding.binding = 0;
        vertexInputBinding.stride = sizeof(Vertex);
		vertexInputBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        std::array<VkVertexInputAttributeDescription, 3> vertexInputAttribs {};
        vertexInputAttribs[0].binding = 0;
        vertexInputAttribs[0].location = 0;
        vertexInputAttribs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        vertexInputAttribs[0].offset = offsetof(Vertex, pos);

        vertexInputAttribs[1].binding = 0;
        vertexInputAttribs[1].location = 1;
        vertexInputAttribs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        vertexInputAttribs[1].offset = offsetof(Vertex, color);

        vertexInputAttribs[2].binding = 0;
        vertexInputAttribs[2].location = 2;
        vertexInputAttribs[2].format = VK_FORMAT_R32G32_SFLOAT;
        vertexInputAttribs[2].offset = offsetof(Vertex, texCoord);

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttribs.size());
        vertexInputInfo.pVertexBindingDescriptions = &vertexInputBinding;
        vertexInputInfo.pVertexAttributeDescriptions = vertexInputAttribs.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f; // Optional
        depthStencil.maxDepthBounds = 1.0f; // Optional
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.front = {}; // Optional
        depthStencil.back = {}; // Optional

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 0;

        VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout));

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;

        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = nullptr; // Optional
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;

        pipelineInfo.layout = pipelineLayout;

        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;

        pipelineInfo.pDepthStencilState = &depthStencil;

        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
        pipelineInfo.basePipelineIndex = -1; // Optional

        VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline));

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }    

    void updateUniformBuffer(uint32_t frame) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo{};
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), windowWidth / (float)windowHeight, 0.1f, 10.0f);
        // glm is originally for OpenGL, whose y coord of the clip space is inverted
        ubo.proj[1][1] *= -1;
        memcpy(uniformBuffers[frame].mapped, &ubo, sizeof(ubo));
    }

    

public:

    void initVulkan() {
        // addExtensions();
        VulkanBase::initVulkan();
    }

    void prepare() {
		VulkanBase::prepare();
		createSyncObjects();
        createDepthStencil();
        createFrameBuffers();
		createCommandBuffers();
		createVertexBuffer();
        createIndexBuffer();
		createUniformBuffers();
		createDescriptorSetLayout();
		createDescriptorPool();
        createTextureImage();
		createDescriptorSets();
		createPipeline();
		prepared = true;
	}

    void cleanUp() {
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyImageView(device, texture.view, nullptr);
        vkDestroySampler(device, texture.sampler, nullptr);
        vmaDestroyImage(allocator, texture.image, texture.alloc);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        for (auto i = 0; i < uniformBuffers.size(); i++) {
            vmaUnmapMemory(allocator, uniformBuffers[i].alloc);
            vmaDestroyBuffer(allocator, uniformBuffers[i].buffer, uniformBuffers[i].alloc);
        }
        vmaDestroyBuffer(allocator, indices.buffer, indices.alloc);
        vmaDestroyBuffer(allocator, vertices.buffer, vertices.alloc);
        vkDestroyCommandPool(device, commandPool, nullptr);
        for (size_t i = 0; i < maxFrameCount; i++) {
            vkDestroySemaphore(device, presentCompleteSemaphores[i], nullptr);
            vkDestroySemaphore(device, renderCompleteSemaphores[i], nullptr);
            vkDestroyFence(device, waitFences[i], nullptr);
        }
        VulkanBase::cleanUp();
        
    }

    void render() {
        if (!prepared)
            return;
        vkWaitForFences(device, 1, &waitFences[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, vulkanSwapchain.swapchain, UINT64_MAX, presentCompleteSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
        
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return;
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        // Set fence to unsignaled
        // Must delay this to after recreateSwapChain to avoid deadlock
        vkResetFences(device, 1, &waitFences[currentFrame]);

        const VkCommandBuffer commandBuffer = commandBuffers[currentFrame];

        vkResetCommandBuffer(commandBuffer, 0);
        VkCommandBufferBeginInfo cmdBufBeginInfo { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        VK_CHECK(vkBeginCommandBuffer(commandBuffer, &cmdBufBeginInfo));

        VkRenderPassBeginInfo renderPassBeginInfo { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
        renderPassBeginInfo.renderPass = renderPass;
        renderPassBeginInfo.framebuffer = swapChainFramebuffers[imageIndex];

        renderPassBeginInfo.renderArea.offset = { 0, 0 };
        renderPassBeginInfo.renderArea.extent = {windowWidth, windowHeight};

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
        clearValues[1].depthStencil = { 1.0f, 0 };

        renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassBeginInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(windowWidth);
        viewport.height = static_cast<float>(windowHeight);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = {windowWidth, windowHeight};
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        VkBuffer vertexBuffers[] = { vertices.buffer };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(commandBuffer, indices.buffer, 0, VK_INDEX_TYPE_UINT16);

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indicesData.size()), 1, 0, 0, 0);
        
        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }

        updateUniformBuffer(currentFrame);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        // Which semaphores to wait
        VkSemaphore waitSemaphores[] = { presentCompleteSemaphores[currentFrame] };
        // On which stage to wait
        // Here we want to wait with writing colors to the image until it's available
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

        // Which semaphores to signal once the command buffer(s) has finished execution
        VkSemaphore signalSemaphores[] = { renderCompleteSemaphores[currentFrame] };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, waitFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = { vulkanSwapchain.swapchain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        presentInfo.pResults = nullptr; // Optional

        result = vkQueuePresentKHR(graphicsQueue, &presentInfo);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }
        currentFrame = (currentFrame + 1) % maxFrameCount;
    }
};

int main() {
    auto app = std::make_unique<VulkanTriangle>();
    app->createWindow();
    app->initVulkan();
    app->prepare();
    app->renderLoop();
    app->cleanUp();
    std::cout << "Completed!!\n";
}