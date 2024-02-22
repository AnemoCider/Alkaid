#include "VulkanBase.h"
#include "utils/VulkanBuffer.h"
#include "asset/VulkanAsset.h"

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

const std::string shaderPathNoSuffix = "../shaders/basicTriangle/basicTriangle";

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;
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

class VulkanTriangle : public Base {

private:
    bool framebufferResized = false;

    vk::DescriptorSetLayout descriptorSetLayout;
    std::vector<vk::DescriptorSet> descriptorSets;
    
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline graphicsPipeline;

    vki::Buffer vertexBuffer;
    vki::Buffer indexBuffer;

    struct {
		vk::DeviceMemory mem;
		vk::Buffer buffer;
		uint32_t count{ 0 };
	} indices;

    std::vector<vki::UniformBuffer> uniformBuffers;

    /*struct ShaderData {
		glm::mat4 projectionMatrix;
		glm::mat4 modelMatrix;
		glm::mat4 viewMatrix;
	};*/

    struct Texture {
        vk::Sampler sampler;
        vk::Image image;
        vk::ImageLayout imageLayout;
        vk::DeviceMemory mem;
        vk::ImageView view;
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
    
    void createDescriptorSetLayout() {
        vk::DescriptorSetLayoutBinding uboLayoutBinding {};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
        uboLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;

        vk::DescriptorSetLayoutBinding textureLayoutBinding{};
        textureLayoutBinding.binding = 1;
        textureLayoutBinding.descriptorCount = 1;
        textureLayoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
        textureLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{ uboLayoutBinding , textureLayoutBinding };
        vk::DescriptorSetLayoutCreateInfo layoutInfo { 
            .bindingCount = static_cast<uint32_t>(setLayoutBindings.size()),
            .pBindings = setLayoutBindings.data()
        };
        
        descriptorSetLayout = device.getDevice().createDescriptorSetLayout(layoutInfo, nullptr);
    }

    void createDescriptorSets() {
        std::vector<vk::DescriptorSetLayout> setLayouts(drawCmdBuffers.size(), descriptorSetLayout);
        vk::DescriptorSetAllocateInfo descriptorSetAI {
            .descriptorPool = descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(descriptorSets.size()),
            .pSetLayouts = setLayouts.data()
        };
        descriptorSets = device.getDevice().allocateDescriptorSets(descriptorSetAI);

        for (size_t i = 0; i < drawCmdBuffers.size(); i++) {
            vk::DescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i].buffer;
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            vk::DescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = texture.imageLayout;
            imageInfo.imageView = texture.view;
            imageInfo.sampler = texture.sampler;

            std::array<vk::WriteDescriptorSet, 2> descriptorWrites{};
            descriptorWrites[0] = {
                .dstSet = descriptorSets[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eUniformBuffer, 
                .pImageInfo = nullptr,
                .pBufferInfo = &bufferInfo
            };
            descriptorWrites[1] = {
                .dstSet = descriptorSets[i],
                .dstBinding = 1,
                .dstArrayElement = 1,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler, 
                .pImageInfo = &imageInfo,
                .pBufferInfo = nullptr
            };
            device.getDevice().updateDescriptorSets(
                static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }

    void createVertexBuffer() {
        /*vk::DeviceSize bufferSize = sizeof(verticesData[0]) * verticesData.size();

        vki::StagingBuffer verticesStaging;

        auto bufferCI = vki::StagingBuffer::getCI(bufferSize);

        verticesStaging.buffer =  device.getDevice().createBuffer(bufferCI);
        auto memReq = device.getDevice().getBufferMemoryRequirements(verticesStaging.buffer);
        vk::MemoryAllocateInfo memAI {
            .allocationSize = memReq.size,
            .memoryTypeIndex =  device.getMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)
        };
        verticesStaging.mem = device.getDevice().allocateMemory(memAI);

        void* data;

        data = device.getDevice().mapMemory(verticesStaging.mem, 0, bufferSize);
        memcpy(data, verticesData.data(), (size_t)bufferSize);
        device.getDevice().unmapMemory(verticesStaging.mem);
        
        vertexBuffer.buffer = device.getDevice().createBuffer(
            bufferCI.setUsage(vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst)
        );
        memReq = device.getDevice().getBufferMemoryRequirements(vertexBuffer.buffer);
        memAI.setAllocationSize(memReq.size);
        memAI.setMemoryTypeIndex(device.getMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal));

        vki::Buffer::copyBuffer(verticesStaging.buffer, vertexBuffer.buffer, bufferSize);

        device.getDevice().freeMemory(verticesStaging.mem);
        device.getDevice().destroyBuffer(verticesStaging.buffer);*/
        vk::DeviceSize bufferSize = sizeof(verticesData[0]) * verticesData.size();
        vki::StagingBuffer staging{ device, bufferSize };

        void* data;

        data = device.getDevice().mapMemory(staging.mem, 0, bufferSize);
        memcpy(data, verticesData.data(), (size_t)bufferSize);
        device.getDevice().unmapMemory(staging.mem);

        vertexBuffer = vki::Buffer{device, bufferSize, 
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst, 
            vk::MemoryPropertyFlagBits::eDeviceLocal };

        staging.clear(device);
    }

    void createIndexBuffer() {
        vk::DeviceSize bufferSize = sizeof(verticesData[0]) * verticesData.size();
        vki::StagingBuffer staging{ device, bufferSize };

        void* data;

        data = device.getDevice().mapMemory(staging.mem, 0, bufferSize);
        memcpy(data, verticesData.data(), (size_t)bufferSize);
        device.getDevice().unmapMemory(staging.mem);

        indexBuffer = vki::Buffer{ device, bufferSize,
            vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal };

        staging.clear(device);
    }

    void createUniformBuffers() {
        vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
        uniformBuffers.resize(drawCmdBuffers.size());

        for (size_t i = 0; i < uniformBuffers.size(); i++) {
            /*uniformBuffers[i].buffer = device.getDevice().createBuffer(bufferCI);
            auto memReqs = device.getDevice().getBufferMemoryRequirements(uniformBuffers[i].buffer);
            auto index = device.getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
            memAI.setAllocationSize(memReqs.size);
            memAI.setMemoryTypeIndex(index);
            uniformBuffers[i].mem = device.getDevice().allocateMemory(memAI);
            uniformBuffers[i].mapped = device.getDevice().mapMemory(uniformBuffers[i].mem, 0, bufferSize);*/
            uniformBuffers[i] = vki::UniformBuffer(device, bufferSize);
        }

    }

    void createTextureImage() {
        std::string filename = getAssetPath() + "textures/metalplate01_rgba.ktx";
        // image format
        vk::Format format = vk::Format::eR8G8B8A8Unorm;

        ktxResult result;
        ktxTexture* ktxTexture;
        result = ktxTexture_CreateFromNamedFile(filename.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktxTexture);
        assert(result == KTX_SUCCESS);
        texture.width = ktxTexture->baseWidth;
        texture.height = ktxTexture->baseHeight;
        texture.mipLevels = ktxTexture->numLevels;
        ktx_uint8_t* ktxTextureData = ktxTexture_GetData(ktxTexture);
        ktx_size_t ktxTextureSize = ktxTexture_GetSize(ktxTexture);

        vki::StagingBuffer staging{device, ktxTextureSize};

        /*createBuffer(ktxTextureSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT, stagingBuffer.buffer, stagingBuffer.alloc);*/

        // Copy texture data into host local staging buffer
        void* data;
        /*VK_CHECK(vmaMapMemory(allocator, stagingBuffer.alloc, &data));
        memcpy(data, ktxTextureData, ktxTextureSize);
        vmaUnmapMemory(allocator, stagingBuffer.alloc);*/
        data = device.getDevice().mapMemory(staging.mem, 0, ktxTextureSize);
        memcpy(data, ktxTextureData, ktxTextureSize);
        device.getDevice().unmapMemory(staging.mem);

        // Setup buffer copy regions for each mip level
        std::vector<vk::BufferImageCopy> bufferCopyRegions;

        for (uint32_t i = 0; i < texture.mipLevels; i++) {
            // Calculate offset into staging buffer for the current mip level
            ktx_size_t offset;
            KTX_error_code ret = ktxTexture_GetImageOffset(ktxTexture, i, 0, 0, &offset);
            assert(ret == KTX_SUCCESS);
            // Setup a buffer image copy structure for the current mip level
            vk::BufferImageCopy bufferCopyRegion = {
                .bufferOffset = offset,
                .imageSubresource = {vk::ImageAspectFlagBits::eColor, i, 0, 1},
                .imageExtent = {ktxTexture->baseWidth >> i, ktxTexture->baseHeight >> i, 1},
            };
            /*bufferCopyRegion.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
            bufferCopyRegion.imageSubresource.mipLevel = i;
            bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
            bufferCopyRegion.imageSubresource.layerCount = 1;*/
            /*bufferCopyRegion.imageExtent.width = ktxTexture->baseWidth >> i;
            bufferCopyRegion.imageExtent.height = ktxTexture->baseHeight >> i;
            bufferCopyRegion.imageExtent.depth = 1;
            bufferCopyRegion.bufferOffset = offset;*/
            bufferCopyRegions.push_back(bufferCopyRegion);
        }

        // Create optimal tiled target image on the device
        /*vk::ImageCreateInfo imageCreateInfo = vki::init_image_create_info(
            VK_IMAGE_TYPE_2D, format, { texture.width, texture.height, 1 }, texture.mipLevels,
            VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);*/

        vk::ImageCreateInfo imageCI{
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent = { texture.width, texture.height, 1 },
            .mipLevels = texture.mipLevels,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined
        };

        texture.image = device.getDevice().createImage(imageCI);
        auto memReqs = device.getDevice().getImageMemoryRequirements(texture.image);

        vk::MemoryAllocateInfo memAI{
            .allocationSize = ktxTextureSize,
            .memoryTypeIndex = device.getMemoryType(memReqs.size, vk::MemoryPropertyFlagBits::eDeviceLocal)
        };

        texture.mem = device.getDevice().allocateMemory(memAI);
        device.getDevice().bindImageMemory(texture.image, texture.mem, ktxTextureSize);

        // Image memory barriers for the texture image

        // The sub resource range describes the regions of the image that will be transitioned using the memory barriers below
        vk::ImageSubresourceRange subresourceRange = {
            // Image only contains color data
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            // Start at first mip level
            .baseMipLevel = 0,
            // We will transition on all mip levels
            .levelCount = texture.mipLevels,
            // The 2D texture only has one layer
            .layerCount = 1
        };

        // Transition the texture image layout to transfer target, so we can safely copy our buffer data to it.
        vk::ImageMemoryBarrier imageMemoryBarrier = {
            .srcAccessMask = vk::AccessFlagBits::eNone,
            .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .image = texture.image,
            .subresourceRange = subresourceRange,
        };

        vk::CommandBuffer commandBuffer = beginSingleTimeCommands();
        // Insert a memory dependency at the proper pipeline stages that will execute the image layout transition
        // Source pipeline stage is host write/read execution (VK_PIPELINE_STAGE_HOST_BIT)
        // Destination pipeline stage is copy command execution (VK_PIPELINE_STAGE_TRANSFER_BIT)
        commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eHost,
            vk::PipelineStageFlagBits::eTransfer,
            vk::DependencyFlags(),
            0, nullptr,
            0, nullptr,
            1, &imageMemoryBarrier
        );

        // Copy mip levels from staging buffer
        commandBuffer.copyBufferToImage(
            staging.buffer,
            texture.image,
            vk::ImageLayout::eTransferDstOptimal,
            static_cast<uint32_t>(bufferCopyRegions.size()),
            bufferCopyRegions.data()
        );

        
        // Once the data has been uploaded we transfer to the texture image to the shader read layout, so it can be sampled from
        imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        imageMemoryBarrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        imageMemoryBarrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

        // Insert a memory dependency at the proper pipeline stages that will execute the image layout transition
        // Source pipeline stage is copy command execution (VK_PIPELINE_STAGE_TRANSFER_BIT)
        // Destination pipeline stage fragment shader access (VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
        commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eFragmentShader,
            vk::DependencyFlags(),
            0, nullptr,
            0, nullptr,
            1, &imageMemoryBarrier
        );

        endSingleTimeCommands(commandBuffer);
        // Store current layout for later reuse
        texture.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

        // Clean up staging resources
        /*vmaDestroyBuffer(allocator, stagingBuffer.buffer, stagingBuffer.alloc);*/
        staging.clear(device);

        ktxTexture_Destroy(ktxTexture);

        // Create a texture sampler
        // In Vulkan textures are accessed by samplers
        // This separates all the sampling information from the texture data. This means you could have multiple sampler objects for the same texture with different settings
        // Note: Similar to the samplers available with OpenGL 3.3
        vk::SamplerCreateInfo samplerCI{
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eLinear,
            .addressModeU = vk::SamplerAddressMode::eRepeat,
            .addressModeV = vk::SamplerAddressMode::eRepeat,
            .addressModeW = vk::SamplerAddressMode::eRepeat,
            .mipLodBias = 0.0f,
            // Enable anisotropic filtering
            // This feature is optional, so we must check if it's supported on the device
            // TODO: Check it
            // Use max. level of anisotropy for this example
            .anisotropyEnable = vk::True,
            .maxAnisotropy = instance.supports.properties.limits.maxSamplerAnisotropy,
            .compareOp = vk::CompareOp::eNever,
            .minLod = 0.0f,
            // Set max level-of-detail to mip level count of the texture
            .maxLod = (float)texture.mipLevels,
            .borderColor = vk::BorderColor::eFloatOpaqueWhite
        };

        texture.sampler = device.getDevice().createSampler(samplerCI);

        // Create image view
        // Textures are not directly accessed by the shaders and
        // are abstracted by image views containing additional
        // information and sub resource ranges
        vk::ImageViewCreateInfo viewCI {
            .image = texture.image,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            // The subresource range describes the set of mip levels (and array layers) that can be accessed through this image view
            // It's possible to create multiple image views for a single image referring to different (and/or overlapping) ranges of the image
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 0, 1, texture.mipLevels}
        };

        texture.view = device.getDevice().createImageView(viewCI);
    }

    void createPipeline() {
        auto shaderCode = vki::readFile(shaderPathNoSuffix + ".vert.spv");
        vk::ShaderModuleCreateInfo shaderCI{
            .codeSize = shaderCode.size(),
            .pCode = reinterpret_cast<const uint32_t*>(shaderCode.data())
        };

        vk::ShaderModule vertShaderModule = device.getDevice().createShaderModule(
            shaderCI
        );
        
        shaderCode = vki::readFile(shaderPathNoSuffix + ".frag.spv");
        shaderCI.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());
        vk::ShaderModule fragShaderModule = device.getDevice().createShaderModule(
            shaderCI
        );
        
        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = vertShaderModule,
            .pName = "name"
        };
            
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = fragShaderModule,
            .pName = "name"
        };

        vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        vk::VertexInputBindingDescription vertexInputBinding {
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = vk::VertexInputRate::eVertex
        };

        std::array<vk::VertexInputAttributeDescription, 3> vertexInputAttribs {
            vk::VertexInputAttributeDescription {
                .location = 0, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, pos)
            }, 
            vk::VertexInputAttributeDescription {
                .location = 1, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, color)
            },
            vk::VertexInputAttributeDescription {
                .location = 2, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(Vertex, texCoord)
            }
        };

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &vertexInputBinding,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttribs.size()),
            .pVertexAttributeDescriptions = vertexInputAttribs.data()
        };

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = vk::PrimitiveTopology::eTriangleList,
            .primitiveRestartEnable = vk::False
        };

        vk::PipelineViewportStateCreateInfo viewportState{
            .viewportCount = 1,
            .scissorCount = 1
        };

        vk::PipelineRasterizationStateCreateInfo rasterizer{
            .depthClampEnable = vk::False,
            .rasterizerDiscardEnable = vk::False,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eBack,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable = vk::False,
            .lineWidth = 1.0f,
        };

        vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = vk::True,
            .depthWriteEnable = vk::True,
            .depthCompareOp = vk::CompareOp::eLess,
            .stencilTestEnable = vk::False
        };
        
        vk::PipelineMultisampleStateCreateInfo multisampling{
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable = vk::False
        };

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = vk::False
        };

        vk::PipelineColorBlendStateCreateInfo colorBlending{
            .logicOpEnable = vk::False
        };

        std::vector<vk::DynamicState> dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };

        vk::PipelineDynamicStateCreateInfo dynamicState{
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data()
        };

        vk::PipelineLayoutCreateInfo pipelineLayoutCI {
            .setLayoutCount = 1,
            .pSetLayouts = &descriptorSetLayout,
            .pushConstantRangeCount = 0
        };

        pipelineLayout = device.getDevice().createPipelineLayout(pipelineLayoutCI);

        vk::GraphicsPipelineCreateInfo pipelineCI {
            .stageCount = 2,
            .pStages = shaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil,
            .pColorBlendState = &colorBlending,
            .pDynamicState = &dynamicState,
            .layout = pipelineLayout,
            .renderPass = renderPass,
            .subpass = 0
        };

        graphicsPipeline = device.getDevice().createGraphicsPipeline(nullptr, pipelineCI).value;

        device.getDevice().destroyShaderModule(fragShaderModule);
        device.getDevice().destroyShaderModule(vertShaderModule);
    }    

    void updateUniformBuffer(uint32_t frame) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo{};
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), instance.width / (float)instance.height, 0.1f, 10.0f);
        // glm is originally for OpenGL, whose y coord of the clip space is inverted
        ubo.proj[1][1] *= -1;
        memcpy(uniformBuffers[frame].mapped, &ubo, sizeof(ubo));
    }

    

public:

    void initVulkan() {
        // addExtensions();
        Base::init();
    }

    void prepare() {
		Base::prepare();
		createVertexBuffer();
        createIndexBuffer();
		createUniformBuffers();
		createDescriptorSetLayout();
        createTextureImage();
		createDescriptorSets();
		createPipeline();
	}

    void clear() {
        device.getDevice().destroyPipelineLayout(pipelineLayout);
        device.getDevice().destroyPipeline(graphicsPipeline);
        device.getDevice().destroyImageView(texture.view);
        device.getDevice().destroySampler(texture.sampler);
        device.getDevice().destroyImage(texture.image);
        device.getDevice().destroyDescriptorSetLayout(descriptorSetLayout);
        for (auto i = 0; i < uniformBuffers.size(); i++) {
            uniformBuffers[i].clear(device);
        }
        indexBuffer.clear(device);
        vertexBuffer.clear(device);
        device.getDevice().destroyCommandPool(commandPool);
        Base::clear();
    }

    void render() {
        device.getDevice().waitForFences(1, &fences[currentBuffer], VK_TRUE, UINT64_MAX);
        Base::prepareFrame();

        // Set fence to unsignaled
        // Must delay this to after recreateSwapChain to avoid deadlock
        device.getDevice().resetFences(1, &fences[currentBuffer]);

        const auto& commandBuffer = drawCmdBuffers[currentBuffer];

        commandBuffer.reset();

        vk::CommandBufferBeginInfo cmdBufBeginInfo{};
        commandBuffer.begin(cmdBufBeginInfo);

        vk::RenderPassBeginInfo renderPassBeginInfo { 
            .renderPass = renderPass,
            .framebuffer = frameBuffers[currentBuffer],
            .renderArea {.offset = { 0, 0 }, .extent = {instance.width, instance.height}}
        };

        std::array<vk::ClearValue, 2> clearValues{};
        clearValues[0].color.setFloat32({ 0.0f, 0.0f, 0.0f, 1.0f });
        clearValues[1].depthStencil = { 1.0f, 0 };

        renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassBeginInfo.pClearValues = clearValues.data();

        commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

        vk::Viewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(instance.width);
        viewport.height = static_cast<float>(instance.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        commandBuffer.setViewport(0, viewport);

        vk::Rect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = { instance.width, instance.height };
        commandBuffer.setScissor(0, scissor);

        vk::Buffer vertexBuffers[] = { vertexBuffer.buffer };
        vk::DeviceSize offsets[] = { 0 };
        commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);

        commandBuffer.bindIndexBuffer(indexBuffer.buffer, 0, vk::IndexType::eUint16);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSets[currentBuffer], 0, nullptr);
        commandBuffer.drawIndexed(static_cast<uint32_t>(indicesData.size()), 1, 0, 0, 0);

        commandBuffer.endRenderPass();
        commandBuffer.end();

        updateUniformBuffer(currentBuffer);

        vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
        vk::SubmitInfo submitInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &semaphores.presentComplete,
            .pWaitDstStageMask = waitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &semaphores.renderComplete
        };
        
        // On which stage to wait
        // Here we want to wait with writing colors to the image until it's available
       

        // Which semaphores to signal once the command buffer(s) has finished execution
        graphicsQueue.submit(submitInfo, fences[currentBuffer]);

        vk::PresentInfoKHR presentInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &semaphores.renderComplete,
            .swapchainCount = 1,
            .pSwapchains = &swapChain.getSwapChain(),
            .pImageIndices = &currentBuffer
        };
        graphicsQueue.presentKHR(presentInfo);

        /*if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }*/
        /*currentFrame = (currentFrame + 1) % maxFrameCount;*/
    }
};

int main() {
    std::unique_ptr<Base> app = std::make_unique<VulkanTriangle>();
    app->init();
    app->prepare();
    app->renderLoop();
    app->clear();
    std::cout << "Completed!!\n";
}