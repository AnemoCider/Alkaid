#include "VulkanBase.h"
#include "utils/VulkanBuffer.h"
#include "asset/VulkanAsset.h"

#include <ktx.h>
#include <ktxvulkan.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyObj/tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <memory>
#include <iostream>
#include <array>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <chrono>

const std::string shaderPath = "shaders/shadowMapping/";

const uint32_t shadowMapWidth = 2048, shadowMapHeight = 2048;

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 texCoord;
    glm::vec3 diffuse;
    glm::vec3 specular;
    float shininess;
    int illum;
};

std::vector<Vertex> floorVertices = {};
std::vector<Vertex> characterVertices = {};

struct alignas(16) UniformBufferObject {
     glm::mat4 model;
     glm::mat4 view;
     glm::mat4 proj;
     glm::mat4 normalRot;
     glm::vec4 lightPos;
     glm::vec4 viewPos;
     glm::mat4 lightVP;
};

struct alignas(16) LightUbo {
    glm::mat4 lightMVP;
};


void loadObj(const std::string& inputfile, const std::string matPath, std::vector<Vertex>& data) {
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = matPath; // Path to material files

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(inputfile, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();

    // loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {

            auto& material = materials[shapes[s].mesh.material_ids[f]];

            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
            
            // loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // assuming triangles

                Vertex curVertex{};
                curVertex.diffuse = { material.diffuse[0], material.diffuse[1], material.diffuse[2] };
                curVertex.specular = { material.specular[0], material.specular[1], material.specular[2] };
                curVertex.shininess = material.shininess;
                curVertex.illum = material.illum;

                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
                curVertex.pos = { vx, vy, vz };

                // check if `normal_index` is zero or positive. negative = no normal data
                if (idx.normal_index >= 0) {
                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                    curVertex.normal = { nx, ny, nz };
                }

                // check if `texcoord_index` is zero or positive. negative = no texcoord data
                if (idx.texcoord_index >= 0) {
                    tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                    curVertex.texCoord = { tx, 1 - ty };
                }

                data.push_back(std::move(curVertex));
                // optional: vertex colors
                // tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
                // tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
                // tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
            }
            index_offset += fv;
        }
    }
}

class Example : public Base {

private:

    bool framebufferResized = false;

    vk::DescriptorSetLayout descriptorSetLayout;
    std::vector<vk::DescriptorSet> descriptorSets;
    vk::DescriptorSetLayout offscreenDescSetLayout;
    std::vector<vk::DescriptorSet> offscreenDescSets;

    vk::PipelineLayout pipelineLayout;
    vk::Pipeline graphicsPipeline;

    vki::Buffer characterVertBuffer;
    vki::Buffer floorVertBuffer;

    struct {
        vk::DeviceMemory mem;
        vk::Buffer buffer;
        uint32_t count{ 0 };
    } indices;

    std::vector<vki::UniformBuffer> uniformBuffers;
    std::vector<vki::UniformBuffer> lightUbos;

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

    std::vector<Texture> textures;

    vki::Camera light;
    Texture shadowMap;
    vk::Format shadowMapFormat = vk::Format::eD16Unorm;

    vk::Framebuffer offscreenFrameBuffer;
    vk::RenderPass offscreenPass;
    vk::PipelineLayout offscreenPipelineLayout;
    vk::Pipeline offscreenPipeline;

    std::string getShaderPathName() {
        return "shaders/shadowMapping/shadowMapping";
    }
    std::string getAssetPath() {
        return "assets/202/";
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<Example*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
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

    void createOffscreenFrameBuffer() {
        vk::ImageView attachment = shadowMap.view;
        vk::FramebufferCreateInfo frameBufferCI{
            .renderPass = offscreenPass,
            .attachmentCount = 1,
            .pAttachments = &attachment,
            .width = shadowMap.width,
            .height = shadowMap.height,
            .layers = 1
        };
        offscreenFrameBuffer = device.getDevice().createFramebuffer(frameBufferCI, nullptr);
    }

    void createDescriptorPool() override {
        textures.resize(2);
        std::vector<vk::DescriptorPoolSize> poolSizes(2);
        poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
        // One for screen, one for offscreen
        poolSizes[0].descriptorCount = static_cast<uint32_t>(drawCmdBuffers.size() * 2);
        poolSizes[1].type = vk::DescriptorType::eCombinedImageSampler;
        // each corresponding to one texture, plus one shadowMap
        poolSizes[1].descriptorCount = static_cast<uint32_t>(drawCmdBuffers.size() * (textures.size() + 1));

        vk::DescriptorPoolCreateInfo poolInfo{
            .maxSets = static_cast<uint32_t>(drawCmdBuffers.size() * 2),
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

        vk::DescriptorSetLayoutCreateInfo layoutInfo{
            .bindingCount = 1,
            .pBindings = &uboLayoutBinding
        };

        offscreenDescSetLayout = device.getDevice().createDescriptorSetLayout(layoutInfo, nullptr);

        vk::DescriptorSetLayoutBinding textureLayoutBinding{};
        textureLayoutBinding.binding = 1;
        textureLayoutBinding.descriptorCount = static_cast<uint32_t>(textures.size());
        textureLayoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
        textureLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

        vk::DescriptorSetLayoutBinding shadowMapLayoutBinding{
            .binding = 2,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment
        };

        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{ uboLayoutBinding , textureLayoutBinding, shadowMapLayoutBinding};
        layoutInfo
            .setBindingCount(static_cast<uint32_t>(setLayoutBindings.size()))
            .setPBindings(setLayoutBindings.data());

        descriptorSetLayout = device.getDevice().createDescriptorSetLayout(layoutInfo, nullptr);
    }

    void createDescriptorSets() {
        std::vector<vk::DescriptorSetLayout> setLayouts(drawCmdBuffers.size(), descriptorSetLayout);
        std::vector<vk::DescriptorSetLayout> offscreenSetLayouts(drawCmdBuffers.size(), offscreenDescSetLayout);
        vk::DescriptorSetAllocateInfo descriptorSetAI{
            .descriptorPool = descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(setLayouts.size()),
            .pSetLayouts = setLayouts.data()
        };

        descriptorSets = device.getDevice().allocateDescriptorSets(descriptorSetAI);
        
        descriptorSetAI.setPSetLayouts(offscreenSetLayouts.data());
        offscreenDescSets = device.getDevice().allocateDescriptorSets(descriptorSetAI);

        std::vector<vk::DescriptorImageInfo> imageInfos(textures.size());
        for (int i = 0; i < textures.size(); i++) {
            imageInfos[i] = {
                .sampler = textures[i].sampler,
                .imageView = textures[i].view,
                .imageLayout = textures[i].imageLayout
            };
        }
        vk::DescriptorImageInfo shadowMapInfo{
            .sampler = shadowMap.sampler,
            .imageView = shadowMap.view,
            .imageLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal
        };
        
        for (size_t i = 0; i < drawCmdBuffers.size(); i++) {
            vk::DescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i].buffer;
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            std::array<vk::WriteDescriptorSet, 3> descriptorWrites{};
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
                .dstArrayElement = 0,
                .descriptorCount = static_cast<uint32_t>(imageInfos.size()),
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = imageInfos.data(),
                .pBufferInfo = nullptr
            };
            descriptorWrites[2] = {
                .dstSet = descriptorSets[i],
                .dstBinding = 2,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &shadowMapInfo, 
                .pBufferInfo = nullptr
            };
            device.getDevice().updateDescriptorSets(
                static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);

            bufferInfo.setBuffer(lightUbos[i].buffer).setRange(sizeof(LightUbo));
            descriptorWrites[0].setDstSet(offscreenDescSets[i]).setPBufferInfo(&bufferInfo);
            device.getDevice().updateDescriptorSets(
                1, &descriptorWrites[0], 0, nullptr);
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

    void createUniformBuffers() {
        vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
        uniformBuffers.resize(drawCmdBuffers.size());

        for (size_t i = 0; i < uniformBuffers.size(); i++) {
            uniformBuffers[i] = vki::UniformBuffer(device, bufferSize);
        }

        bufferSize = sizeof(LightUbo);
        lightUbos.resize(drawCmdBuffers.size());

        for (size_t i = 0; i < lightUbos.size(); i++) {
            lightUbos[i] = vki::UniformBuffer(device, bufferSize);
        }

    }

    void createShadowMap() {
        shadowMap.width = shadowMapWidth;
        shadowMap.height = shadowMapHeight;

        vk::ImageCreateInfo imageCI{
            .imageType = vk::ImageType::e2D,
            .format = shadowMapFormat,
            .extent = {shadowMap.width, shadowMap.height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled

        };
        shadowMap.image = device.getDevice().createImage(imageCI, nullptr);

        auto memReq = device.getDevice().
            getImageMemoryRequirements(shadowMap.image);

        vk::MemoryAllocateInfo memAI{
            .allocationSize = memReq.size,
            .memoryTypeIndex = device.getMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)
        };
        shadowMap.mem = device.getDevice().allocateMemory(memAI);
        device.getDevice().bindImageMemory(shadowMap.image, shadowMap.mem, 0);

        shadowMap.imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

        vk::SamplerCreateInfo samplerCI{
            .magFilter = vk::Filter::eNearest,
            .minFilter = vk::Filter::eNearest,
            .mipmapMode = vk::SamplerMipmapMode::eLinear,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToBorder,
            .addressModeW = vk::SamplerAddressMode::eClampToBorder,
            .mipLodBias = 0.0f,
            .anisotropyEnable = vk::False,
            .maxAnisotropy = 1.0f,
            .minLod = 0.0f,
            // Set max level-of-detail to mip level count of the texture
            .maxLod = 1.0f,
            .borderColor = vk::BorderColor::eFloatOpaqueWhite
        };

        shadowMap.sampler = device.getDevice().createSampler(samplerCI);

        vk::ImageSubresourceRange range{
            .aspectMask = vk::ImageAspectFlagBits::eDepth,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };

        vk::ImageViewCreateInfo imageViewCI{
            .image = shadowMap.image,
            .viewType = vk::ImageViewType::e2D,
            .format = shadowMapFormat,
            .subresourceRange = range
        };
        //// Stencil aspect should only be set on depth + stencil formats (VK_FORMAT_D16_UNORM_S8_UINT..VK_FORMAT_D32_SFLOAT_S8_UINT
        //if (instance.depthFormat >= vk::Format::eD16UnormS8Uint) {
        //    imageViewCI.subresourceRange.aspectMask |= vk::ImageAspectFlagBits::eStencil;
        //}
        shadowMap.view = device.getDevice().createImageView(imageViewCI);
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

    void createOffscreenPass() {
        vk::AttachmentDescription depthDescription{
            .format = shadowMapFormat,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined,
            .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal
        };

        vk::AttachmentReference depthRef{
            .attachment = 0,
            .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal
        };

        vk::SubpassDescription subPass{
            .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
            .inputAttachmentCount = 0,
            .pInputAttachments = nullptr,
            .colorAttachmentCount = 0,
            .pColorAttachments = nullptr,
            .pResolveAttachments = nullptr,
            .pDepthStencilAttachment = &depthRef,
            .preserveAttachmentCount = 0,
            .pPreserveAttachments = nullptr,
        };

        vk::SubpassDependency dependency {
            .srcSubpass = vk::SubpassExternal,
            .dstSubpass = 0,
            .srcStageMask = 
                vk::PipelineStageFlagBits::eFragmentShader,
            .dstStageMask =
                vk::PipelineStageFlagBits::eEarlyFragmentTests,
            .srcAccessMask =
                vk::AccessFlagBits::eShaderRead,
            .dstAccessMask =
                vk::AccessFlagBits::eDepthStencilAttachmentWrite,
        };

        vk::RenderPassCreateInfo renderPassCI{
            .attachmentCount = 1,
            .pAttachments = &depthDescription,
            .subpassCount = 1,
            .pSubpasses = &subPass,
            .dependencyCount = 1,
            .pDependencies = &dependency
        };

        offscreenPass = device.getDevice().createRenderPass(renderPassCI, nullptr);
    }

    void createPipeline() {
        auto shaderCode = vki::readFile(shaderPath + "offscreen.vert.spv");
        vk::ShaderModuleCreateInfo shaderCI{
            .codeSize = shaderCode.size(),
            .pCode = reinterpret_cast<const uint32_t*>(shaderCode.data())
        };

        vk::ShaderModule vertShaderModule = device.getDevice().createShaderModule(
            shaderCI
        );

        shaderCode = vki::readFile(shaderPath + "offscreen.frag.spv");
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
        std::array<vk::VertexInputAttributeDescription, 7> vertexInputAttribs{
            vk::VertexInputAttributeDescription {
                .location = 0, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, pos)
            },
            vk::VertexInputAttributeDescription {
                .location = 1, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, normal)
            },
            vk::VertexInputAttributeDescription {
                .location = 2, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(Vertex, texCoord)
            },
            vk::VertexInputAttributeDescription {
                .location = 3, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, diffuse)
            },
            vk::VertexInputAttributeDescription {
                .location = 4, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, specular)
            },
            vk::VertexInputAttributeDescription {
                .location = 5, .binding = 0, .format = vk::Format::eR32Sfloat, .offset = offsetof(Vertex, shininess)
            },
            vk::VertexInputAttributeDescription {
                .location = 6, .binding = 0, .format = vk::Format::eR32Sint, .offset = offsetof(Vertex, illum)
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
            .pSetLayouts = &offscreenDescSetLayout,
            .pushConstantRangeCount = 0
        };

        offscreenPipelineLayout = device.getDevice().createPipelineLayout(pipelineLayoutCI);

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
            .layout = offscreenPipelineLayout,
            .renderPass = offscreenPass,
            .subpass = 0
        };

        offscreenPipeline = device.getDevice().createGraphicsPipeline(nullptr, pipelineCI).value;

        device.getDevice().destroyShaderModule(fragShaderModule);
        device.getDevice().destroyShaderModule(vertShaderModule);

        shaderCode = vki::readFile(shaderPath + "screen.vert.spv");
        shaderCI.setCodeSize(shaderCode.size()).setPCode(reinterpret_cast<const uint32_t*>(shaderCode.data()));

        vertShaderModule = device.getDevice().createShaderModule(
            shaderCI
        );

        shaderCode = vki::readFile(shaderPath + "screen.frag.spv");
        shaderCI.codeSize = shaderCode.size();
        shaderCI.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());
        fragShaderModule = device.getDevice().createShaderModule(
            shaderCI
        );

        vertShaderStageInfo.setModule(vertShaderModule);
        fragShaderStageInfo.setModule(fragShaderModule);

        shaderStages[0] = vertShaderStageInfo;
        shaderStages[1] = fragShaderStageInfo;

        vk::PushConstantRange pushConstantRange {
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
            .offset = 0,
            .size = sizeof(uint32_t)
        };

        pipelineLayoutCI
            .setPSetLayouts(&descriptorSetLayout)
            .setPushConstantRangeCount(1)
            .setPPushConstantRanges(&pushConstantRange);

        pipelineLayout = device.getDevice().createPipelineLayout(pipelineLayoutCI);

        pipelineCI.layout = pipelineLayout;
        pipelineCI.renderPass = renderPass;
        graphicsPipeline = device.getDevice().createGraphicsPipeline(nullptr, pipelineCI).value;

        device.getDevice().destroyShaderModule(fragShaderModule);
        device.getDevice().destroyShaderModule(vertShaderModule);
    }

    void updateUniformBuffer(uint32_t frame) {
        light.position = { 10.0f, 15.0f, 10.0f};
        light.zoom = 100.f;
        /*camera.position = light.position;
        camera.front = glm::normalize(glm::vec3(0.0f, 1.5f, 0.0f) - camera.position);*/
        UniformBufferObject ubo{};
        ubo.model = glm::mat4(1.0f);
        ubo.normalRot = glm::mat3(glm::transpose(glm::inverse(ubo.model)));
        ubo.view = camera.view();
        ubo.proj = camera.projection((float)instance.width, (float)instance.height);
        ubo.lightPos = { light.position, 1.0f };
        ubo.viewPos = { camera.position, 1.0f };
        // glm is originally for OpenGL, whose y coord of the clip space is inverted
        ubo.proj[1][1] *= -1;

        auto proj = light.projection((float)shadowMapWidth, (float)shadowMapHeight, 10.0f, 200.0f);
        proj[1][1] *= -1;

        LightUbo lightUbo{
            .lightMVP = 
                proj
                * glm::lookAt(light.position, {0.0f, 1.5f, 0.0f}, light.up) 
                * glm::mat4(1.0f)
        };

        ubo.lightVP = lightUbo.lightMVP;
        memcpy(uniformBuffers[frame].mapped, &ubo, sizeof(ubo));
        memcpy(lightUbos[frame].mapped, &lightUbo, sizeof(lightUbo));
    }

    void buildCommandBuffer() override {
        const auto& commandBuffer = drawCmdBuffers[currentBuffer];

        commandBuffer.reset();

        vk::CommandBufferBeginInfo cmdBufBeginInfo{};
        commandBuffer.begin(cmdBufBeginInfo);

        vk::RenderPassBeginInfo renderPassBeginInfo{
            .renderPass = offscreenPass,
            .framebuffer = offscreenFrameBuffer,
            .renderArea {.offset = { 0, 0 }, .extent = {shadowMap.width, shadowMap.height}}
        };

        std::array<vk::ClearValue, 2> clearValues{};
        clearValues[0].depthStencil = { 1.0f, 0 };

        renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassBeginInfo.pClearValues = clearValues.data();

        commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, offscreenPipeline);

        vk::Viewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(shadowMapWidth);
        viewport.height = static_cast<float>(shadowMapHeight);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        commandBuffer.setViewport(0, viewport);

        vk::Rect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = { shadowMapWidth, shadowMapHeight };
        commandBuffer.setScissor(0, scissor);

        vk::Buffer vertexBuffers[] = { characterVertBuffer.buffer };
        vk::DeviceSize offsets[] = { 0 };

        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, offscreenPipelineLayout, 0, 1, &offscreenDescSets[currentBuffer], 0, nullptr);
        commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);
        commandBuffer.draw(static_cast<uint32_t>(characterVertices.size()), 1, 0, 0);

        vertexBuffers[0] = floorVertBuffer.buffer;
        commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);
        commandBuffer.draw(static_cast<uint32_t>(floorVertices.size()), 1, 0, 0);

        commandBuffer.endRenderPass();

        vk::ImageMemoryBarrier imageMemoryBarrier = {
            .srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .oldLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            .newLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal,
            .image = shadowMap.image,
            .subresourceRange {
                .aspectMask = vk::ImageAspectFlagBits::eDepth,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            }
        };

        commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests,
            vk::PipelineStageFlagBits::eFragmentShader,
            vk::DependencyFlags(),
            nullptr,
            nullptr,
            imageMemoryBarrier
        );

        clearValues[0].color.setFloat32({0.0f, 0.0f, 0.0f, 1.0f});
        clearValues[1].depthStencil = vk::ClearDepthStencilValue({1.0f, 0});

        renderPassBeginInfo.setRenderPass(renderPass)
            .setFramebuffer(frameBuffers[currentBuffer])
            .setRenderArea({ .offset = { 0, 0 }, .extent = {instance.width, instance.height} })
            .setClearValueCount(clearValues.size())
            .setPClearValues(clearValues.data());

        commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
        viewport.setWidth(instance.width).setHeight(instance.height);
        scissor.setExtent({ instance.width, instance.height });
        commandBuffer.setViewport(0, viewport);
        commandBuffer.setScissor(0, scissor);

        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSets[currentBuffer], 0, nullptr);
        vertexBuffers[0] = characterVertBuffer.buffer;
        commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);
        commandBuffer.pushConstants<uint32_t>(pipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, (uint32_t)0);
        commandBuffer.draw(static_cast<uint32_t>(characterVertices.size()), 1, 0, 0);
        vertexBuffers[0] = floorVertBuffer.buffer;
        commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);
        commandBuffer.pushConstants<uint32_t>(pipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, (uint32_t)1);
        commandBuffer.draw(static_cast<uint32_t>(floorVertices.size()), 1, 0, 0);

        commandBuffer.endRenderPass();

        commandBuffer.end();
    }

public:

    void prepare() override {
        Base::prepare();
        camera = vki::Camera{ -5.0f, 1.5f, 0.0f };
        glfwSetFramebufferSizeCallback(instance.window, framebufferResizeCallback);
        glfwSetScrollCallback(instance.window, scroll_callback);
        glfwSetMouseButtonCallback(instance.window, mouse_button_callback);
        glfwSetCursorPosCallback(instance.window, mouse_callback);
        loadObj(getAssetPath() + "marry/Marry.obj", getAssetPath() + "marry/", characterVertices);
        loadObj(getAssetPath() + "floor/floor.obj", getAssetPath() + "floor/", floorVertices);
        createVertexBuffer(characterVertices, characterVertBuffer);
        createVertexBuffer(floorVertices, floorVertBuffer);
        createUniformBuffers();
        createDescriptorSetLayout();
        createTextureImage(getAssetPath() + "marry/MC003_Kozakura_Mari.png", textures[0]);
        createTextureImage(getAssetPath() + "floor/floor.png", textures[1]);
        createShadowMap();
        createOffscreenPass();
        createOffscreenFrameBuffer();
        createDescriptorSets();
        createPipeline();
    }

    void clear() override {
        device.getDevice().destroyPipelineLayout(offscreenPipelineLayout);
        device.getDevice().destroyPipeline(offscreenPipeline);
        device.getDevice().destroyPipelineLayout(pipelineLayout);
        device.getDevice().destroyPipeline(graphicsPipeline);
        device.getDevice().destroyFramebuffer(offscreenFrameBuffer);
        device.getDevice().destroyRenderPass(offscreenPass);
        destroyTexture(shadowMap);
        destroyTexture(textures[1]);
        destroyTexture(textures[0]);
        device.getDevice().destroyDescriptorSetLayout(offscreenDescSetLayout);
        device.getDevice().destroyDescriptorSetLayout(descriptorSetLayout);
        for (auto i = 0; i < uniformBuffers.size(); i++) {
            device.getDevice().unmapMemory(lightUbos[i].mem);
            lightUbos[i].clear(device);
            device.getDevice().unmapMemory(uniformBuffers[i].mem);
            uniformBuffers[i].clear(device);
        }
        characterVertBuffer.clear(device);
        floorVertBuffer.clear(device);
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