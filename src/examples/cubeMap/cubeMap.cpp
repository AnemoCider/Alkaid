#include "VulkanBase.h"
#include "utils/VulkanBuffer.h"
#include "asset/VulkanAsset.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"

#include <memory>
#include <iostream>
#include <array>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <chrono>

const std::string shaderFolder = "shaders/cubeMap/";

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 texCoord;
};

std::vector<Vertex> verticesData = {};
std::vector<uint16_t> indicesData = {};

struct alignas(16) UniformBufferObject {
     glm::mat4 model;
     glm::mat4 view;
     glm::mat4 proj;
     glm::mat4 normalRot;
     glm::vec4 lightPos;
     glm::vec4 viewPos;
};


class Example : public Base {

private:

    tinygltf::TinyGLTF loader;

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
    };

    std::string getAssetPath() {
        return "Vulkan-Assets/";
    }

    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
        auto app = reinterpret_cast<Example*>(glfwGetWindowUserPointer(window));
        app->camera.zoomIn(static_cast<float>(yoffset));
    }
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
        auto app = reinterpret_cast<Example*>(glfwGetWindowUserPointer(window));
        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
            app->camera.startDrag();
        else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
            app->camera.disableDrag();
    }
    static void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
        auto app = reinterpret_cast<Example*>(glfwGetWindowUserPointer(window));
        app->camera.mouseDrag(static_cast<float>(xpos), static_cast<float>(ypos));
    }

    decltype(auto) loadGlTFModel(const std::string& path) {
        tinygltf::Model model;
        std::string err;
        std::string warn;
        auto ret = loader.LoadASCIIFromFile(&model, &err, &warn, path);
        if (!warn.empty()) {
            std::cout << "TinyglTF [warning] when loading model: " <<
                warn.c_str() << '\n';
        }
        if (!err.empty()) {
            std::cout << "TinyglTF [error] when loading model: " <<
                err.c_str() << '\n';
        }
        assert(ret);
        return model;
    }

    void loadAssets() {
        auto model = loadGlTFModel(getAssetPath() + "models/sphere.gltf");
        // there is only one mesh and one primitive
        // the properties (and their indices)
        auto& primitive = model.meshes[0].primitives[0];
        const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];

        // Indices
        const tinygltf::BufferView& bufferView = model.bufferViews[indexAccessor.bufferView];
        const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
        const uint16_t* indicesBuffer = reinterpret_cast<const uint16_t*>(&buffer.data[bufferView.byteOffset + indexAccessor.byteOffset]);
        indicesData.insert(indicesData.end(), indicesBuffer, indicesBuffer + indexAccessor.count);

        // Get the accessor indices for position, normal, and texcoords
        int posAccessorIndex = primitive.attributes.find("POSITION")->second;
        int normAccessorIndex = primitive.attributes.find("NORMAL")->second;
        int texCoordAccessorIndex = primitive.attributes.find("TEXCOORD_0")->second;

        // Obtain the buffer views corresponding to the accessors
        const tinygltf::Accessor& posAccessor = model.accessors[posAccessorIndex];
        const tinygltf::Accessor& normAccessor = model.accessors[normAccessorIndex];
        const tinygltf::Accessor& texCoordAccessor = model.accessors[texCoordAccessorIndex];

        // Obtain buffer data for positions, normals, and texcoords
        const tinygltf::BufferView& posView = model.bufferViews[posAccessor.bufferView];
        const tinygltf::BufferView& normView = model.bufferViews[normAccessor.bufferView];
        const tinygltf::BufferView& texCoordView = model.bufferViews[texCoordAccessor.bufferView];

        verticesData.reserve(posAccessor.count);

        const float* bufferPos = reinterpret_cast<const float*>(&(model.buffers[posView.buffer].data[posAccessor.byteOffset + posView.byteOffset]));
        const float* bufferNormals = reinterpret_cast<const float*>(&(model.buffers[normView.buffer].data[normAccessor.byteOffset + normView.byteOffset]));
        const float* bufferTexCoords = reinterpret_cast<const float*>(&(model.buffers[texCoordView.buffer].data[texCoordAccessor.byteOffset + texCoordView.byteOffset]));

        for (size_t v = 0; v < posAccessor.count; v++) {
            Vertex vert{};

            vert.pos = glm::vec4(glm::make_vec3(&bufferPos[v * 3]), 1.0f);
            vert.normal = glm::normalize(glm::make_vec3(&bufferNormals[v * 3]));
            vert.texCoord = glm::make_vec2(&bufferTexCoords[v * 2]);
            verticesData.emplace_back(std::move(vert));
        }
        // loadCubeMap(getAssetPath() + "textures/cubemap_yokohama_rgba.ktx")
    }

    void createDescriptorPool() override {
        std::vector<vk::DescriptorPoolSize> poolSizes(1);
        poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(drawCmdBuffers.size());

        vk::DescriptorPoolCreateInfo poolInfo{
            .maxSets = static_cast<uint32_t>(drawCmdBuffers.size()),
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()
        };
        descriptorPool = device.getDevice().createDescriptorPool(poolInfo);
    };

    void createDescriptorSetLayout() {
        vk::DescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
        uboLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;

        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{ uboLayoutBinding };
        vk::DescriptorSetLayoutCreateInfo layoutInfo{
            .bindingCount = static_cast<uint32_t>(setLayoutBindings.size()),
            .pBindings = setLayoutBindings.data()
        };

        descriptorSetLayout = device.getDevice().createDescriptorSetLayout(layoutInfo, nullptr);
    }

    void createDescriptorSets() {
        std::vector<vk::DescriptorSetLayout> setLayouts(drawCmdBuffers.size(), descriptorSetLayout);
        vk::DescriptorSetAllocateInfo descriptorSetAI{
            .descriptorPool = descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(setLayouts.size()),
            .pSetLayouts = setLayouts.data()
        };
        descriptorSets = device.getDevice().allocateDescriptorSets(descriptorSetAI);
        
        for (size_t i = 0; i < drawCmdBuffers.size(); i++) {
            vk::DescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i].buffer;
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            std::array<vk::WriteDescriptorSet, 1> descriptorWrites{};
            descriptorWrites[0] = {
                .dstSet = descriptorSets[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .pImageInfo = nullptr,
                .pBufferInfo = &bufferInfo
            };
            device.getDevice().updateDescriptorSets(
                static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }

    void createVertexBuffer(const std::vector<Vertex>& vertices, vki::Buffer& dstBuffer) {
        vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
        vki::StagingBuffer staging{ device, bufferSize };

        void* data;

        data = device.getDevice().mapMemory(staging.mem, 0, bufferSize);
        memcpy(data, vertices.data(), (size_t)bufferSize);
        device.getDevice().unmapMemory(staging.mem);

        dstBuffer = vki::Buffer{ device, bufferSize,
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal };

        copyBuffer(staging.buffer, dstBuffer.buffer, bufferSize);
        staging.clear(device);
    }

    void createIndexBuffer() {
        vk::DeviceSize bufferSize = sizeof(indicesData[0]) * indicesData.size();
        vki::StagingBuffer staging{ device, bufferSize };

        void* data;

        data = device.getDevice().mapMemory(staging.mem, 0, bufferSize);
        memcpy(data, indicesData.data(), (size_t)bufferSize);
        device.getDevice().unmapMemory(staging.mem);

        indexBuffer = vki::Buffer{ device, bufferSize,
            vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal };
        copyBuffer(staging.buffer, indexBuffer.buffer, bufferSize);
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

    void createTextureImage(const std::string& file, Texture& texture) {
        int width, height, nrChannels;
        unsigned char* textureData = stbi_load(file.c_str(), &width, &height, &nrChannels, STBI_rgb_alpha);
        assert(textureData);
        // 4 bytes a pixel: R8G8B8A8
        auto bufferSize = width * height * 4;
        texture.width = width;
        texture.height = height;
        texture.mipLevels = 1;
        // image format
        vk::Format format = vk::Format::eR8G8B8A8Unorm;
        
        vki::StagingBuffer staging{ device, static_cast<vk::DeviceSize>(bufferSize) };

        void* data;

        data = device.getDevice().mapMemory(staging.mem, 0, bufferSize);
        memcpy(data, textureData, bufferSize);
        device.getDevice().unmapMemory(staging.mem);


        vk::BufferImageCopy bufferCopyRegion = {
            .bufferOffset = 0,
            .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .imageExtent = {texture.width, texture.height, 1},
        };

        vk::ImageCreateInfo imageCI{
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent = { texture.width, texture.height, 1 },
            .mipLevels = texture.mipLevels,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined
        };

        texture.image = device.getDevice().createImage(imageCI);
        auto memReqs = device.getDevice().getImageMemoryRequirements(texture.image);

        vk::MemoryAllocateInfo memAI{
            .allocationSize = memReqs.size,
            .memoryTypeIndex = device.getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)
        };

        texture.mem = device.getDevice().allocateMemory(memAI);
        device.getDevice().bindImageMemory(texture.image, texture.mem, 0);

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
            .baseArrayLayer = 0,
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
            1,
            &bufferCopyRegion
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

        stbi_image_free(textureData);

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

        vk::ImageViewCreateInfo viewCI{
            .image = texture.image,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            // The subresource range describes the set of mip levels (and array layers) that can be accessed through this image view
            // It's possible to create multiple image views for a single image referring to different (and/or overlapping) ranges of the image
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = texture.mipLevels,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        texture.view = device.getDevice().createImageView(viewCI);
    }

    void destroyTexture(Texture& texture) {
        device.getDevice().destroyImageView(texture.view);
        device.getDevice().destroySampler(texture.sampler);
        device.getDevice().freeMemory(texture.mem);
        device.getDevice().destroyImage(texture.image);
    }

    void createPipeline() {
        auto shaderCode = vki::readFile(shaderFolder + "cubeMap.vert.spv");
        vk::ShaderModuleCreateInfo shaderCI{
            .codeSize = shaderCode.size(),
            .pCode = reinterpret_cast<const uint32_t*>(shaderCode.data())
        };

        vk::ShaderModule vertShaderModule = device.getDevice().createShaderModule(
            shaderCI
        );

        shaderCode = vki::readFile(shaderFolder + "cubeMap.frag.spv");
        shaderCI.codeSize = shaderCode.size();
        shaderCI.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());
        vk::ShaderModule fragShaderModule = device.getDevice().createShaderModule(
            shaderCI
        );

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = vertShaderModule,
            .pName = "main"
        };

        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = fragShaderModule,
            .pName = "main"
        };

        vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        vk::VertexInputBindingDescription vertexInputBinding{
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = vk::VertexInputRate::eVertex
        };
        /*
            glm::vec3 pos;
            glm::vec3 normal;
            glm::vec2 texCoord;
            glm::vec3 diffuse;
            glm::vec3 specular;
            float shininess;
            int illum;
        */
        std::array<vk::VertexInputAttributeDescription, 3> vertexInputAttribs{
            vk::VertexInputAttributeDescription {
                .location = 0, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, pos)
            },
            vk::VertexInputAttributeDescription {
                .location = 1, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, normal)
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
            .cullMode = vk::CullModeFlagBits::eBack ,
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
            .blendEnable = vk::False,
            .colorWriteMask =
            vk::ColorComponentFlagBits::eR |
            vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB |
            vk::ColorComponentFlagBits::eA
        };

        vk::PipelineColorBlendStateCreateInfo colorBlending{
            .logicOpEnable = vk::False,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment
        };

        std::vector<vk::DynamicState> dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };

        vk::PipelineDynamicStateCreateInfo dynamicState{
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data()
        };

        vk::PipelineLayoutCreateInfo pipelineLayoutCI{
            .setLayoutCount = 1,
            .pSetLayouts = &descriptorSetLayout,
            .pushConstantRangeCount = 0
        };

        pipelineLayout = device.getDevice().createPipelineLayout(pipelineLayoutCI);

        vk::GraphicsPipelineCreateInfo pipelineCI{
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
        UniformBufferObject ubo{};
        ubo.model = glm::mat4(1.0f);
        ubo.normalRot = glm::mat3(glm::transpose(glm::inverse(ubo.model)));
        ubo.view = camera.view();
        ubo.proj = camera.projection((float)instance.width, (float)instance.height);
        ubo.lightPos = { 5.0f, 5.0f, 5.0f, 0.0f};
        ubo.viewPos = { camera.position, 0.0f };
        // glm is originally for OpenGL, whose y coord of the clip space is inverted
        ubo.proj[1][1] *= -1;
        memcpy(uniformBuffers[frame].mapped, &ubo, sizeof(ubo));
    }

    void buildCommandBuffer() override {
        const auto& commandBuffer = drawCmdBuffers[currentBuffer];

        commandBuffer.reset();

        vk::CommandBufferBeginInfo cmdBufBeginInfo{};
        commandBuffer.begin(cmdBufBeginInfo);

        vk::RenderPassBeginInfo renderPassBeginInfo{
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

        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSets[currentBuffer], 0, nullptr);
        commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);
        commandBuffer.bindIndexBuffer(indexBuffer.buffer, 0, vk::IndexType::eUint16);
        commandBuffer.drawIndexed(static_cast<uint32_t>(indicesData.size()), 1, 0, 0, 0);

        commandBuffer.endRenderPass();
        commandBuffer.end();
    }

public:

    void prepare() override {
        Base::prepare();
        camera = vki::Camera{ -5.0f, 1.5f, 0.0f };
        glfwSetScrollCallback(instance.window, scroll_callback);
        glfwSetMouseButtonCallback(instance.window, mouse_button_callback);
        glfwSetCursorPosCallback(instance.window, mouse_callback);
        loadAssets();
        createVertexBuffer(verticesData, vertexBuffer);
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorSetLayout();
        createDescriptorSets();
        createPipeline();
    }

    void clear() override {
        device.getDevice().destroyPipelineLayout(pipelineLayout);
        device.getDevice().destroyPipeline(graphicsPipeline);
        device.getDevice().destroyDescriptorSetLayout(descriptorSetLayout);
        for (auto i = 0; i < uniformBuffers.size(); i++) {
            device.getDevice().unmapMemory(uniformBuffers[i].mem);
            uniformBuffers[i].clear(device);
        }
        indexBuffer.clear(device);
        vertexBuffer.clear(device);
        // device.getDevice().destroyCommandPool(commandPool);
        Base::clear();
    }

    void render() override {
        static int count = 0;
        auto result = device.getDevice().waitForFences(1, &fences[currentBuffer], VK_TRUE, UINT64_MAX);
        assert(result == vk::Result::eSuccess);
        Base::prepareFrame();

        // Set fence to unsignaled
        // Must delay this to after recreateSwapChain to avoid deadlock
        result = device.getDevice().resetFences(1, &fences[currentBuffer]);
        assert(result == vk::Result::eSuccess);

        buildCommandBuffer();

        camera.update(instance.window);
        updateUniformBuffer(currentBuffer);

        vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
        vk::SubmitInfo submitInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &semaphores.presentComplete,
            .pWaitDstStageMask = waitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = &drawCmdBuffers[currentBuffer],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &semaphores.renderComplete
        };

        graphicsQueue.submit(submitInfo, fences[currentBuffer]);
        Base::presentFrame();
    }
};

int main() {
    std::unique_ptr<Base> app = std::make_unique<Example>();
    app->init();
    app->prepare();
    app->renderLoop();
    app->clear();
    std::cout << "Completed!!\n";
}