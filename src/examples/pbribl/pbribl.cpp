#include "VulkanBase.h"
#include "utils/VulkanBuffer.h"
#include "asset/VulkanAsset.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"

#include <ktx.h>
#include <ktxvulkan.h>

#include <memory>
#include <iostream>
#include <array>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <chrono>

const std::string shaderFolder = "shaders/pbribl/";

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 texCoord;
};

struct CubeVertex {
    glm::vec3 pos;
};

std::vector<Vertex> verticesData = {};
std::vector<uint16_t> indicesData = {};

std::vector<CubeVertex> skyBoxVertices = {};
std::vector<uint16_t> skyBoxIndices = {};

struct alignas(16) UniformBufferObject {
     glm::mat4 model;
     glm::mat4 view;
     glm::mat4 proj;
     glm::mat4 normalRot;
     glm::vec4 viewPos;
};

struct alignas(16) SkyBoxUbo {
    glm::mat4 view;
    glm::mat4 proj;
};


class Example : public Base {

private:

    tinygltf::TinyGLTF loader;

    vk::DescriptorSetLayout descriptorSetLayout;
    std::vector<vk::DescriptorSet> descriptorSets;

    vk::DescriptorSetLayout skyBoxDescLayout;
    std::vector<vk::DescriptorSet> skyBoxDescSets;

    vk::PipelineLayout pipelineLayout;
    vk::Pipeline graphicsPipeline;

    vk::PipelineLayout skyBoxPipelineLayout;
    vk::Pipeline skyBoxPipeline;

    vki::Buffer vertexBuffer;
    vki::Buffer indexBuffer;

    vki::Buffer skyBoxVertexBuffer;
    vki::Buffer skyBoxIndexBuffer;

    struct {
        vk::DeviceMemory mem;
        vk::Buffer buffer;
        uint32_t count{ 0 };
    } indices;

    std::vector<vki::UniformBuffer> uniformBuffers;
    std::vector<vki::UniformBuffer> skyBoxUniformBuffers;

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


    vk::Format cubeMapFormat = vk::Format::eR8G8B8A8Unorm;

    Texture cubeMap;

    struct {
        Texture colorMap;
        vk::Framebuffer frameBuffer;
        vk::RenderPass renderPass;
        vk::Pipeline pipeline;
    } irradiance;

    struct MatPushBlock {
        float roughness = 0.541931f;
        float metallic = 0.496791f;
        float specular = 0.449419f;
        float r = 1.0f;
        float g = 1.0f;
        float b = 1.0f;
    };

    struct {
        Texture table;
    } lut;

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

    void loadModel() {
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

            vert.pos = glm::make_vec3(&bufferPos[v * 3]);
            vert.normal = glm::normalize(glm::make_vec3(&bufferNormals[v * 3]));
            vert.texCoord = glm::make_vec2(&bufferTexCoords[v * 2]);
            verticesData.emplace_back(std::move(vert));
        }
    }

    void loadCube() {
        auto model = loadGlTFModel(getAssetPath() + "models/cube.gltf");
        // there is only one mesh and one primitive
        // the properties (and their indices)
        auto& primitive = model.meshes[0].primitives[0];
        const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];

        // Indices
        const tinygltf::BufferView& bufferView = model.bufferViews[indexAccessor.bufferView];
        const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
        const uint16_t* indicesBuffer = reinterpret_cast<const uint16_t*>(&buffer.data[bufferView.byteOffset + indexAccessor.byteOffset]);
        skyBoxIndices.insert(skyBoxIndices.end(), indicesBuffer, indicesBuffer + indexAccessor.count);

        // Get the accessor indices for position, normal, and texcoords
        int posAccessorIndex = primitive.attributes.find("POSITION")->second;

        // Obtain the buffer views corresponding to the accessors
        const tinygltf::Accessor& posAccessor = model.accessors[posAccessorIndex];

        // Obtain buffer data for positions, normals, and texcoords
        const tinygltf::BufferView& posView = model.bufferViews[posAccessor.bufferView];

        skyBoxVertices.reserve(posAccessor.count);

        const float* bufferPos = reinterpret_cast<const float*>(&(model.buffers[posView.buffer].data[posAccessor.byteOffset + posView.byteOffset]));
        for (size_t v = 0; v < posAccessor.count; v++) {
            CubeVertex vert{};
            vert.pos = glm::make_vec3(&bufferPos[v * 3]);
            skyBoxVertices.emplace_back(std::move(vert));
        }
    }

    void createLut() {
        const vk::Format format = vk::Format::eR16G16Sfloat;
        const int32_t dim = 512;

        vk::ImageCreateInfo imageCI{
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent {
                .width = dim,
                .height = dim,
                .depth = 1
            },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled
        };

        lut.table.image = device.getDevice().createImage(imageCI);

        auto memReqs = device.getDevice().getImageMemoryRequirements(lut.table.image);

        vk::MemoryAllocateInfo memAI{
            .allocationSize = memReqs.size,
            .memoryTypeIndex = device.getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)
        };

        lut.table.mem = device.getDevice().allocateMemory(memAI);
        device.getDevice().bindImageMemory(lut.table.image, lut.table.mem, 0);

        vk::SamplerCreateInfo samplerCI{
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eLinear,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
            .minLod = 0.0f,
            .maxLod = 1.0f,
            .borderColor = vk::BorderColor::eFloatOpaqueWhite
        };

        lut.table.sampler = device.getDevice().createSampler(samplerCI);

        vk::AttachmentDescription colorDescription {
            .format = format,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined,
            .finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal
        };
        vk::AttachmentReference colorRef{
            .attachment = 0,
            .layout = vk::ImageLayout::eColorAttachmentOptimal
        };

        vk::SubpassDescription subPass{
            .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
            .inputAttachmentCount = 0,
            .pInputAttachments = nullptr,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorRef,
            .pResolveAttachments = nullptr,
            .pDepthStencilAttachment = nullptr,
            .preserveAttachmentCount = 0,
            .pPreserveAttachments = nullptr,
        };

        std::array<vk::SubpassDependency, 2> dependencies;
        dependencies[0] = {
            .srcSubpass = vk::SubpassExternal,
            .dstSubpass = 0,
            .srcStageMask = vk::PipelineStageFlagBits::eBottomOfPipe,
            .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .srcAccessMask = vk::AccessFlagBits::eMemoryRead,
            .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eColorAttachmentRead,
            .dependencyFlags = vk::DependencyFlagBits::eByRegion
        };
        dependencies[1] = {
            .srcSubpass = vk::SubpassExternal,
            .dstSubpass = 0,
            .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .dstStageMask = vk::PipelineStageFlagBits::eBottomOfPipe,
            .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eColorAttachmentRead,
            .dstAccessMask = vk::AccessFlagBits::eMemoryRead,
            .dependencyFlags = vk::DependencyFlagBits::eByRegion
        };

        vk::RenderPassCreateInfo renderPassCI{
            .attachmentCount = 1,
            .pAttachments = &colorDescription,
            .subpassCount = 1,
            .pSubpasses = &subPass,
            .dependencyCount = static_cast<uint32_t>(dependencies.size()),
            .pDependencies = dependencies.data()
        };

        auto renderPass = device.getDevice().createRenderPass(renderPassCI, nullptr);

        vk::ImageViewCreateInfo viewCI{
            .image = lut.table.image,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        lut.table.view = device.getDevice().createImageView(viewCI);

        vk::FramebufferCreateInfo frameBufferCI{
            .renderPass = renderPass,
            .attachmentCount = 1,
            .pAttachments = &lut.table.view,
            .width = dim,
            .height = dim,
            .layers = 1
        };

        auto frameBuffer = device.getDevice().createFramebuffer(frameBufferCI);

        auto shaderCode = vki::readFile(shaderFolder + "lut.vert.spv");
        vk::ShaderModuleCreateInfo shaderCI{
            .codeSize = shaderCode.size(),
            .pCode = reinterpret_cast<const uint32_t*>(shaderCode.data())
        };

        vk::ShaderModule vertShaderModule = device.getDevice().createShaderModule(
            shaderCI
        );

        shaderCode = vki::readFile(shaderFolder + "lut.frag.spv");
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

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo = {};

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
            .cullMode = vk::CullModeFlagBits::eNone,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable = vk::False,
            .lineWidth = 1.0f,
        };

        vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = vk::False,
            .depthWriteEnable = vk::False,
            .depthCompareOp = vk::CompareOp::eLessOrEqual ,
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
            .setLayoutCount = 0,
            .pushConstantRangeCount = 0,
        };

        auto pipeLayout = device.getDevice().createPipelineLayout(pipelineLayoutCI);

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
            .layout = pipeLayout,
            .renderPass = renderPass,
            .subpass = 0
        };

        auto pipeline = device.getDevice().createGraphicsPipeline(nullptr, pipelineCI).value;

        vk::ClearValue clearValue{};
        clearValue.color.setFloat32({ 0.0f, 0.0f, 0.0f, 1.0f });

        vk::RenderPassBeginInfo renderPassBeginInfo{
            .renderPass = renderPass,
            .framebuffer = frameBuffer,
            .renderArea {
                .extent {
                    .width = dim,
                    .height = dim
                }
            },
            .clearValueCount = 1,
            .pClearValues = &clearValue
        };

        auto cmdBuffer = beginSingleTimeCommands();

        vk::Viewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(dim);
        viewport.height = static_cast<float>(dim);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        cmdBuffer.setViewport(0, viewport);

        vk::Rect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = { dim, dim };
        cmdBuffer.setScissor(0, scissor);

        cmdBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
        cmdBuffer.setViewport(0, viewport);
        cmdBuffer.setScissor(0, scissor);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
        cmdBuffer.draw(3, 1, 0, 0);
        cmdBuffer.endRenderPass();

        endSingleTimeCommands(cmdBuffer);

        device.getDevice().destroyShaderModule(vertShaderModule);
        device.getDevice().destroyShaderModule(fragShaderModule);
        device.getDevice().destroyPipeline(pipeline);
        device.getDevice().destroyPipelineLayout(pipeLayout);
        device.getDevice().destroyRenderPass(renderPass);
        device.getDevice().destroyFramebuffer(frameBuffer);
    }

    void createIrradianceMap() {
        const uint32_t dim = 512;
        vk::Format format = vk::Format::eR16G16B16A16Sfloat;

        irradiance.colorMap.width = dim;
        irradiance.colorMap.height = dim;

        // roughness r is stored in miplevel of r * (mipCount - 1)
        const uint32_t mipCount = static_cast<uint32_t>(floor(log2(dim))) + 1;
        irradiance.colorMap.mipLevels = mipCount;

        vk::ImageCreateInfo imageCI {
            .flags = vk::ImageCreateFlagBits::eCubeCompatible,
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent = { irradiance.colorMap.width, irradiance.colorMap.height, 1 },
            .mipLevels = irradiance.colorMap.mipLevels,
            .arrayLayers = 6,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined
        };

        irradiance.colorMap.image = device.getDevice().createImage(imageCI);
        auto memReqs = device.getDevice().getImageMemoryRequirements(irradiance.colorMap.image);

        vk::MemoryAllocateInfo memAI{
            .allocationSize = memReqs.size,
            .memoryTypeIndex = device.getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)
        };

        irradiance.colorMap.mem = device.getDevice().allocateMemory(memAI);
        device.getDevice().bindImageMemory(irradiance.colorMap.image, irradiance.colorMap.mem, 0);

        vk::SamplerCreateInfo samplerCI{
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eLinear,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
            .minLod = 0.0f,
            // Set max level-of-detail to mip level count of the texture
            .maxLod = static_cast<float>(mipCount),
            .borderColor = vk::BorderColor::eFloatOpaqueWhite
        };

        irradiance.colorMap.sampler = device.getDevice().createSampler(samplerCI);

        vk::ImageSubresourceRange subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = mipCount,
            .baseArrayLayer = 0,
            .layerCount = 6
        };

        vk::ImageMemoryBarrier imageMemoryBarrier = {
            .srcAccessMask = vk::AccessFlagBits::eNone,
            .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .image = irradiance.colorMap.image,
            .subresourceRange = subresourceRange,
        };

        auto layoutBuffer = beginSingleTimeCommands();

        layoutBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eHost,
            vk::PipelineStageFlagBits::eTransfer,
            vk::DependencyFlags(),
            0, nullptr,
            0, nullptr,
            1, &imageMemoryBarrier
        );

        endSingleTimeCommands(layoutBuffer);

        vk::AttachmentDescription colorDescription{
            .format = format,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined,
            .finalLayout = vk::ImageLayout::eTransferSrcOptimal
        };
        vk::AttachmentReference colorRef{
            .attachment = 0,
            .layout = vk::ImageLayout::eColorAttachmentOptimal
        };

        vk::SubpassDescription subPass{
            .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
            .inputAttachmentCount = 0,
            .pInputAttachments = nullptr,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorRef,
            .pResolveAttachments = nullptr,
            .pDepthStencilAttachment = nullptr,
            .preserveAttachmentCount = 0,
            .pPreserveAttachments = nullptr,
        };

        std::array<vk::SubpassDependency, 2> dependencies;
        dependencies[0] = {
            .srcSubpass = vk::SubpassExternal,
            .dstSubpass = 0,
            .srcStageMask = vk::PipelineStageFlagBits::eBottomOfPipe,
            .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .srcAccessMask = vk::AccessFlagBits::eMemoryRead,
            .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eColorAttachmentRead,
            .dependencyFlags = vk::DependencyFlagBits::eByRegion
        };
        dependencies[1] = {
            .srcSubpass = vk::SubpassExternal,
            .dstSubpass = 0,
            .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .dstStageMask = vk::PipelineStageFlagBits::eBottomOfPipe,
            .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eColorAttachmentRead,
            .dstAccessMask = vk::AccessFlagBits::eMemoryRead,
            .dependencyFlags = vk::DependencyFlagBits::eByRegion
        };

        vk::RenderPassCreateInfo renderPassCI{
            .attachmentCount = 1,
            .pAttachments = &colorDescription,
            .subpassCount = 1,
            .pSubpasses = &subPass,
            .dependencyCount = static_cast<uint32_t>(dependencies.size()),
            .pDependencies = dependencies.data()
        };

        irradiance.renderPass = device.getDevice().createRenderPass(renderPassCI, nullptr);

        vk::ImageViewCreateInfo viewCI {
            .image = irradiance.colorMap.image,
            .viewType = vk::ImageViewType::eCube,
            .format = format,
            // The subresource range describes the set of mip levels (and array layers) that can be accessed through this image view
            // It's possible to create multiple image views for a single image referring to different (and/or overlapping) ranges of the image
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = mipCount,
                .baseArrayLayer = 0,
                .layerCount = 6
            }
        };

        irradiance.colorMap.view = device.getDevice().createImageView(viewCI);

        // render one of the six faces in one mipLevel of the cubeMap at a time
        // transfer to the corresponding range
        // repeat
        imageCI = {
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent = { irradiance.colorMap.width, irradiance.colorMap.height, 1 },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eColorAttachment,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined
        };

        auto image = device.getDevice().createImage(imageCI);

        memReqs = device.getDevice().getImageMemoryRequirements(image);

        memAI = {
            .allocationSize = memReqs.size,
            .memoryTypeIndex = device.getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)
        };

        auto imageMem = device.getDevice().allocateMemory(memAI);
        device.getDevice().bindImageMemory(image, imageMem, 0);

        viewCI = {
            .image = image,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            // The subresource range describes the set of mip levels (and array layers) that can be accessed through this image view
            // It's possible to create multiple image views for a single image referring to different (and/or overlapping) ranges of the image
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        auto view = device.getDevice().createImageView(viewCI);

        vk::FramebufferCreateInfo frameBufferCI{
            .renderPass = irradiance.renderPass,
            .attachmentCount = 1,
            .pAttachments = &view,
            .width = irradiance.colorMap.width,
            .height = irradiance.colorMap.height,
            .layers = 1
        };

        irradiance.frameBuffer = device.getDevice().createFramebuffer(frameBufferCI);

        vk::DescriptorSetLayoutBinding binding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment
        };

        vk::DescriptorSetLayoutCreateInfo descLayoutCI{
            .bindingCount = 1,
            .pBindings = &binding
        };

        auto descLayout = device.getDevice().createDescriptorSetLayout(descLayoutCI);

        vk::DescriptorPoolSize poolSize{
            .type = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1
        };

        vk::DescriptorPoolCreateInfo descPoolCI{
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &poolSize,
        };

        auto descPool = device.getDevice().createDescriptorPool(descPoolCI);

        vk::DescriptorSetAllocateInfo descSetAI{
            .descriptorPool = descPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &descLayout
        };

        auto descSets = device.getDevice().allocateDescriptorSets(descSetAI);

        vk::DescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = cubeMap.imageLayout;
        imageInfo.imageView = cubeMap.view;
        imageInfo.sampler = cubeMap.sampler;

        vk::WriteDescriptorSet writeDescSet{
            .dstSet = descSets[0],
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &imageInfo,
            .pBufferInfo = nullptr
        };

        device.getDevice().updateDescriptorSets(1, &writeDescSet, 0, nullptr);


        auto shaderCode = vki::readFile(shaderFolder + "filter.vert.spv");
        vk::ShaderModuleCreateInfo shaderCI{
            .codeSize = shaderCode.size(),
            .pCode = reinterpret_cast<const uint32_t*>(shaderCode.data())
        };

        vk::ShaderModule vertShaderModule = device.getDevice().createShaderModule(
            shaderCI
        );

        shaderCode = vki::readFile(shaderFolder + "filter.frag.spv");
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

        vk::VertexInputBindingDescription vertexInputBinding = {
            .binding = 0,
            .stride = sizeof(CubeVertex),
            .inputRate = vk::VertexInputRate::eVertex
        };

        std::array<vk::VertexInputAttributeDescription, 1> vertexInputAttribs{
            vk::VertexInputAttributeDescription {
                .location = 0, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(CubeVertex, pos)
            }
        };

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo = {
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &vertexInputBinding,
            .vertexAttributeDescriptionCount = 1,
            .pVertexAttributeDescriptions = &vertexInputAttribs[0]
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
            .cullMode = vk::CullModeFlagBits::eNone,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable = vk::False,
            .lineWidth = 1.0f,
        };

        vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = vk::False,
            .depthWriteEnable = vk::False,
            .depthCompareOp = vk::CompareOp::eLessOrEqual ,
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

        struct PushBlock {
            glm::mat4 mvp;
            float roughness;
            uint32_t numSamples = 32;
        } pushBlock;

        vk::PushConstantRange pcRange{
            .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            .offset = 0,
            .size = sizeof(PushBlock)
        };

        vk::PipelineLayoutCreateInfo pipelineLayoutCI{
            .setLayoutCount = 1,
            .pSetLayouts = &descLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pcRange
        };

        auto pipeLayout = device.getDevice().createPipelineLayout(pipelineLayoutCI);

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
            .layout = pipeLayout,
            .renderPass = irradiance.renderPass,
            .subpass = 0
        };

        auto pipeline = device.getDevice().createGraphicsPipeline(nullptr, pipelineCI).value;

        vk::ClearValue clearValue{};
        clearValue.color.setFloat32({ 0.0f, 0.0f, 0.0f, 1.0f });

        vk::RenderPassBeginInfo renderPassBeginInfo{
            .renderPass = irradiance.renderPass,
            .framebuffer = irradiance.frameBuffer,
            .renderArea {
                .extent {
                    .width = irradiance.colorMap.width,
                    .height = irradiance.colorMap.height
                }
            },
            .clearValueCount = 1,
            .pClearValues = &clearValue
        };

        // view matrices
        std::vector<glm::mat4> matrices = {
            // POSITIVE_X
            glm::rotate(glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f)), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
            // NEGATIVE_X
            glm::rotate(glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(0.0f, 1.0f, 0.0f)), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
            // POSITIVE_Y
            glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
            // NEGATIVE_Y
            glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
            // POSITIVE_Z
            glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
            // NEGATIVE_Z
            glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
        };

        auto cmdBuffer = beginSingleTimeCommands();

        vk::Viewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(dim);
        viewport.height = static_cast<float>(dim);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        cmdBuffer.setViewport(0, viewport);

        vk::Rect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = { dim, dim };
        cmdBuffer.setScissor(0, scissor);
        
        for (uint32_t m = 0; m < mipCount; m++) {
            pushBlock.roughness = (float)m / (float)(mipCount - 1);
            for (uint32_t f = 0; f < 6; f++) {
                viewport.width = static_cast<float>(dim * std::pow(0.5f, m));
                viewport.height = static_cast<float>(dim * std::pow(0.5f, m));
                cmdBuffer.setViewport(0, viewport);

                // Render scene from cube face's point of view
                cmdBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

                // Update shader push constant block
                pushBlock.mvp = glm::perspective((float)(3.14159265 / 2.0), 1.0f, 0.1f, 512.0f) * matrices[f];

                cmdBuffer.pushConstants(pipeLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(PushBlock), &pushBlock);

                cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
                cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeLayout, 0, 1, descSets.data(), 0, nullptr);

                cmdBuffer.bindVertexBuffers(0, skyBoxVertexBuffer.buffer, { 0 });
                cmdBuffer.bindIndexBuffer(skyBoxIndexBuffer.buffer, 0, vk::IndexType::eUint16);
                cmdBuffer.drawIndexed(static_cast<uint32_t>(skyBoxIndices.size()), 1, 0, 0, 0);

                cmdBuffer.endRenderPass();

                subresourceRange = {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1
                };

                /*vk::ImageMemoryBarrier imageMemoryBarrier = {
                    .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                    .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                    .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
                    .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                    .image = image,
                    .subresourceRange = subresourceRange,
                };
                
                cmdBuffer.pipelineBarrier(
                    vk::PipelineStageFlagBits::eColorAttachmentOutput,
                    vk::PipelineStageFlagBits::eTransfer,
                    vk::DependencyFlags(),
                    0, nullptr,
                    0, nullptr,
                    1, &imageMemoryBarrier
                );*/

                vk::ImageCopy copyRegion{
                    .srcSubresource {
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .mipLevel = 0,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
                    .srcOffset = {0, 0, 0},
                    .dstSubresource {
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .mipLevel = m,
                        .baseArrayLayer = f,
                        .layerCount = 1,
                    },
                    .dstOffset = {0, 0, 0},
                    .extent {
                        static_cast<uint32_t>(viewport.width),
                        static_cast<uint32_t>(viewport.height),
                        1
                    }
                };
                
                cmdBuffer.copyImage(
                    image,
                    vk::ImageLayout::eTransferSrcOptimal,
                    irradiance.colorMap.image,
                    vk::ImageLayout::eTransferDstOptimal,
                    1,
                    &copyRegion
                );

                vk::ImageMemoryBarrier imageMemoryBarrier = {
                    .srcAccessMask = vk::AccessFlagBits::eTransferRead,
                    .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                    .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                    .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
                    .image = image,
                    .subresourceRange = subresourceRange
                };

                cmdBuffer.pipelineBarrier(
                    vk::PipelineStageFlagBits::eTransfer,
                    vk::PipelineStageFlagBits::eColorAttachmentOutput,
                    vk::DependencyFlags(),
                    0, nullptr,
                    0, nullptr,
                    1, &imageMemoryBarrier
                );
            }
        }
        subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = mipCount,
            .baseArrayLayer = 0,
            .layerCount = 6
        };

        imageMemoryBarrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .image = irradiance.colorMap.image,
            .subresourceRange = subresourceRange,
        };

        cmdBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eFragmentShader,
            vk::DependencyFlags(),
            0, nullptr,
            0, nullptr,
            1, &imageMemoryBarrier
        );

        endSingleTimeCommands(cmdBuffer);


        device.getDevice().destroyPipeline(pipeline);
        device.getDevice().destroyPipelineLayout(pipeLayout);
        device.getDevice().destroyShaderModule(fragShaderModule);
        device.getDevice().destroyShaderModule(vertShaderModule);
        device.getDevice().destroyImageView(view);
        device.getDevice().freeMemory(imageMem);
        device.getDevice().destroyImage(image);
        device.getDevice().destroyDescriptorPool(descPool);
        device.getDevice().destroyDescriptorSetLayout(descLayout);
        device.getDevice().destroyFramebuffer(irradiance.frameBuffer);
        device.getDevice().destroyRenderPass(irradiance.renderPass);

    }

    void createDescriptorPool() override {
        std::vector<vk::DescriptorPoolSize> poolSizes(2);
        poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(drawCmdBuffers.size() * 2);
        poolSizes[1].type = vk::DescriptorType::eCombinedImageSampler;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(drawCmdBuffers.size() * 3);

        vk::DescriptorPoolCreateInfo poolInfo{
            .maxSets = static_cast<uint32_t>(drawCmdBuffers.size() * 2),
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()
        };
        descriptorPool = device.getDevice().createDescriptorPool(poolInfo);
    };

    void createDescriptorSetLayout() {
        vk::DescriptorSetLayoutBinding uboLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex
        };

        vk::DescriptorSetLayoutBinding lutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment
        };

        vk::DescriptorSetLayoutBinding filterBinding{
            .binding = 2,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment
        };

        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{ uboLayoutBinding, lutBinding, filterBinding};
        vk::DescriptorSetLayoutCreateInfo layoutInfo{
            .bindingCount = static_cast<uint32_t>(setLayoutBindings.size()),
            .pBindings = setLayoutBindings.data()
        };

        descriptorSetLayout = device.getDevice().createDescriptorSetLayout(layoutInfo, nullptr);
        skyBoxDescLayout = device.getDevice().createDescriptorSetLayout(layoutInfo, nullptr);
    }

    void createDescriptorSets() {
        std::vector<vk::DescriptorSetLayout> skyBoxSetLayouts(drawCmdBuffers.size(), skyBoxDescLayout);
        vk::DescriptorSetAllocateInfo descriptorSetAI{
            .descriptorPool = descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(skyBoxSetLayouts.size()),
            .pSetLayouts = skyBoxSetLayouts.data()
        };
        skyBoxDescSets = device.getDevice().allocateDescriptorSets(descriptorSetAI);

        std::vector<vk::DescriptorSetLayout> setLayouts(drawCmdBuffers.size(), descriptorSetLayout);
        descriptorSetAI = {
            .descriptorPool = descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(setLayouts.size()),
            .pSetLayouts = setLayouts.data()
        };
        descriptorSets = device.getDevice().allocateDescriptorSets(descriptorSetAI);
        
        for (size_t i = 0; i < drawCmdBuffers.size(); i++) {
            vk::DescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = skyBoxUniformBuffers[i].buffer;
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(SkyBoxUbo);

            vk::DescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = cubeMap.imageLayout;
            imageInfo.imageView = cubeMap.view;
            imageInfo.sampler = cubeMap.sampler;

            std::vector<vk::WriteDescriptorSet> descriptorWrites(3);
            descriptorWrites[0] = {
                .dstSet = skyBoxDescSets[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .pImageInfo = nullptr,
                .pBufferInfo = &bufferInfo
            };
            descriptorWrites[1] = {
                .dstSet = skyBoxDescSets[i],
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &imageInfo,
                .pBufferInfo = nullptr
            };
            device.getDevice().updateDescriptorSets(
                2, descriptorWrites.data(), 0, nullptr);

            bufferInfo.setBuffer(uniformBuffers[i].buffer).setRange(sizeof(UniformBufferObject));
            descriptorWrites[0].setDstSet(descriptorSets[i]).setPBufferInfo(&bufferInfo);

            vk::DescriptorImageInfo lutInfo{
                .sampler = lut.table.sampler,
                .imageView = lut.table.view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            };

            imageInfo = {
                .sampler = irradiance.colorMap.sampler,
                .imageView = irradiance.colorMap.view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            };

            descriptorWrites[1].setDstSet(descriptorSets[i]).setPImageInfo(&lutInfo);
            descriptorWrites[2] = {
                .dstSet = descriptorSets[i],
                .dstBinding = 2,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &imageInfo,
                .pBufferInfo = nullptr
            };

            device.getDevice().updateDescriptorSets(
                static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }

    template<typename T>
    void createVertexBuffer(const std::vector<T>& vertices, vki::Buffer& dstBuffer) {
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

    void createIndexBuffer(const std::vector<uint16_t>& indices, vki::Buffer& dstBuffer) {
        vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();
        vki::StagingBuffer staging{ device, bufferSize };

        void* data;

        data = device.getDevice().mapMemory(staging.mem, 0, bufferSize);
        memcpy(data, indices.data(), (size_t)bufferSize);
        device.getDevice().unmapMemory(staging.mem);

        dstBuffer = vki::Buffer{ device, bufferSize,
            vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal };
        copyBuffer(staging.buffer, dstBuffer.buffer, bufferSize);
        staging.clear(device);
    }

    void createUniformBuffers() {
        vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
        vk::DeviceSize skyBoxBufferSize = sizeof(SkyBoxUbo);
        uniformBuffers.resize(drawCmdBuffers.size());
        skyBoxUniformBuffers.resize(drawCmdBuffers.size());
        for (size_t i = 0; i < uniformBuffers.size(); i++) {
            skyBoxUniformBuffers[i] = vki::UniformBuffer(device, skyBoxBufferSize);
            uniformBuffers[i] = vki::UniformBuffer(device, bufferSize);
        }
    }

    void createCubeMap(const std::string& file, Texture& texture) {
        ktxResult result;
        ktxTexture* ktxTexture;
        result = ktxTexture_CreateFromNamedFile(file.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktxTexture);
        assert(result == KTX_SUCCESS);
        texture.width = ktxTexture->baseWidth;
        texture.height = ktxTexture->baseHeight;
        texture.mipLevels = ktxTexture->numLevels;
        ktx_uint8_t* ktxTextureData = ktxTexture_GetData(ktxTexture);
        ktx_size_t ktxTextureSize = ktxTexture_GetSize(ktxTexture);
        
        vki::StagingBuffer staging{ device, static_cast<vk::DeviceSize>(ktxTextureSize) };

        void* data;

        data = device.getDevice().mapMemory(staging.mem, 0, ktxTextureSize);
        memcpy(data, ktxTextureData, ktxTextureSize);
        device.getDevice().unmapMemory(staging.mem);


        std::vector<vk::BufferImageCopy> bufferCopyRegions;

        for (uint32_t face = 0; face < 6; face++) {
            for (uint32_t level = 0; level < texture.mipLevels; level++) {
                ktx_size_t offset;
                KTX_error_code ret = ktxTexture_GetImageOffset(ktxTexture, level, 0, face, &offset);
                assert(ret == KTX_SUCCESS);
                vk::BufferImageCopy bufferCopyRegion = {
                    .bufferOffset = offset,
                    .imageSubresource {
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .mipLevel = level,
                        .baseArrayLayer = face,
                        .layerCount = 1,
                    },
                    .imageExtent {
                        .width = ktxTexture->baseWidth >> level,
                        .height = ktxTexture->baseHeight >> level,
                        .depth = 1
                    }
                };
                bufferCopyRegions.push_back(bufferCopyRegion);
            }
        }

        vk::ImageCreateInfo imageCI{
            .flags = vk::ImageCreateFlagBits::eCubeCompatible,
            .imageType = vk::ImageType::e2D,
            .format = cubeMapFormat,
            .extent = { texture.width, texture.height, 1 },
            .mipLevels = texture.mipLevels,
            .arrayLayers = 6,
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
            .layerCount = 6
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
            bufferCopyRegions.size(),
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
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
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
            .viewType = vk::ImageViewType::eCube,
            .format = cubeMapFormat,
            // The subresource range describes the set of mip levels (and array layers) that can be accessed through this image view
            // It's possible to create multiple image views for a single image referring to different (and/or overlapping) ranges of the image
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = texture.mipLevels,
                .baseArrayLayer = 0,
                .layerCount = 6
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

        vk::PushConstantRange pcRange{
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
            .offset = 0,
            .size = sizeof(MatPushBlock)
        };

        vk::PipelineLayoutCreateInfo pipelineLayoutCI{
            .setLayoutCount = 1,
            .pSetLayouts = &descriptorSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pcRange
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


        shaderCode = vki::readFile(shaderFolder + "skyBox.vert.spv");
        shaderCI.setCodeSize(shaderCode.size()).setPCode(reinterpret_cast<const uint32_t*>(shaderCode.data()));

        vertShaderModule = device.getDevice().createShaderModule(
            shaderCI
        );

        shaderCode = vki::readFile(shaderFolder + "skyBox.frag.spv");
        shaderCI.setCodeSize(shaderCode.size()).setPCode(reinterpret_cast<const uint32_t*>(shaderCode.data()));

        fragShaderModule = device.getDevice().createShaderModule(
            shaderCI
        );

        vertShaderStageInfo = {
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = vertShaderModule,
            .pName = "main"
        };

        fragShaderStageInfo = {
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = fragShaderModule,
            .pName = "main"
        };

        shaderStages[0] = vertShaderStageInfo;
        shaderStages[1] = fragShaderStageInfo;

        vertexInputBinding = {
            .binding = 0,
            .stride = sizeof(CubeVertex),
            .inputRate = vk::VertexInputRate::eVertex
        };

        vertexInputInfo = {
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &vertexInputBinding,
            .vertexAttributeDescriptionCount = 1,
            .pVertexAttributeDescriptions = &vertexInputAttribs[0]
        };

        rasterizer.setCullMode(vk::CullModeFlagBits::eFront);
        depthStencil.setDepthTestEnable(vk::False).setDepthWriteEnable(vk::False).setDepthCompareOp(vk::CompareOp::eLessOrEqual);
        
        pipelineLayoutCI.setPSetLayouts(&skyBoxDescLayout);

        skyBoxPipelineLayout = device.getDevice().createPipelineLayout(pipelineLayoutCI);

        pipelineCI.setPVertexInputState(&vertexInputInfo)
            .setLayout(skyBoxPipelineLayout);

        skyBoxPipeline = device.getDevice().createGraphicsPipeline(nullptr, pipelineCI).value;

        device.getDevice().destroyShaderModule(fragShaderModule);
        device.getDevice().destroyShaderModule(vertShaderModule);
    }


    void updateUniformBuffer(uint32_t frame) {
        UniformBufferObject ubo{};
        ubo.model = glm::mat4(1.0f);
        ubo.normalRot = glm::mat3(glm::transpose(glm::inverse(ubo.model)));
        ubo.view = camera.view();
        ubo.proj = camera.projection((float)instance.width, (float)instance.height);
        ubo.viewPos = { camera.position, 0.0f };
        // glm is originally for OpenGL, whose y coord of the clip space is inverted
        ubo.proj[1][1] *= -1;
        memcpy(uniformBuffers[frame].mapped, &ubo, sizeof(ubo));

        SkyBoxUbo skyBoxUbo{};
        skyBoxUbo.view = ubo.view;
        skyBoxUbo.proj = ubo.proj;
        // camera translation should not affect skyBox
        skyBoxUbo.view[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        memcpy(skyBoxUniformBuffers[frame].mapped, &skyBoxUbo, sizeof(skyBoxUbo));
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

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, skyBoxPipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, skyBoxPipelineLayout, 0, 1, &skyBoxDescSets[currentBuffer], 0, nullptr);
        commandBuffer.bindVertexBuffers(0, skyBoxVertexBuffer.buffer, offsets);
        commandBuffer.bindIndexBuffer(skyBoxIndexBuffer.buffer, 0, vk::IndexType::eUint16);
        commandBuffer.drawIndexed(static_cast<uint32_t>(skyBoxIndices.size()), 1, 0, 0, 0);

        MatPushBlock material{ .roughness = 0.0f, .metallic = 1.0f };
        commandBuffer.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(MatPushBlock), &material);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
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
        loadModel();
        loadCube();
        createVertexBuffer(verticesData, vertexBuffer);
        createVertexBuffer(skyBoxVertices, skyBoxVertexBuffer);
        createIndexBuffer(indicesData, indexBuffer);
        createIndexBuffer(skyBoxIndices, skyBoxIndexBuffer);
        createUniformBuffers();
        createCubeMap(getAssetPath() + "textures/cubemap_yokohama_rgba.ktx", cubeMap);
        createLut();
        createIrradianceMap();
        createDescriptorSetLayout();
        createDescriptorSets();
        createPipeline();
    }

    void clear() override {
        device.getDevice().destroyPipelineLayout(skyBoxPipelineLayout);
        device.getDevice().destroyPipeline(skyBoxPipeline);
        device.getDevice().destroyPipelineLayout(pipelineLayout);
        device.getDevice().destroyPipeline(graphicsPipeline);
        device.getDevice().destroyDescriptorSetLayout(skyBoxDescLayout);
        device.getDevice().destroyDescriptorSetLayout(descriptorSetLayout);
        destroyTexture(lut.table);
        destroyTexture(irradiance.colorMap);
        destroyTexture(cubeMap);
        for (auto i = 0; i < uniformBuffers.size(); i++) {
            device.getDevice().unmapMemory(skyBoxUniformBuffers[i].mem);
            skyBoxUniformBuffers[i].clear(device);
            device.getDevice().unmapMemory(uniformBuffers[i].mem);
            uniformBuffers[i].clear(device);
        }
        skyBoxIndexBuffer.clear(device);
        skyBoxVertexBuffer.clear(device);
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