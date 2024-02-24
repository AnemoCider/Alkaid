#pragma once

#include "common/VulkanCommon.h"
#include "initialization/VulkanDevice.h"
#include "initialization/VulkanInstance.h"
#include "preparation/VulkanSwapChain.h"
#include "utils/VulkanBuffer.h"
#include "control/VulkanCamera.h"

class Base {

protected:

	vki::Instance instance;
	vki::Device device;
	// Handle to the graphics queue, should be obtained during Base::Init
	vk::Queue graphicsQueue;
	vki::SwapChain swapChain;
	vk::Pipeline pipeline;
	vki::Camera camera;

	vk::CommandPool commandPool;
	// Command buffers used for rendering
	std::vector<vk::CommandBuffer> drawCmdBuffers;
	// Global render pass for frame buffer writes
	vk::RenderPass renderPass{ nullptr };
	struct ImageData{
		vk::Image image;
		vk::DeviceMemory mem;
		vk::ImageView view;
	};
	// By default, usage = depthStencil attachment, no mip
	ImageData depthStencil;
	// List of available frame buffers (same as number of swap chain images)
	std::vector<vk::Framebuffer>frameBuffers;
	// Active frame buffer index, updated by acquireNextImage
	uint32_t currentBuffer = 0;
	struct {
		// Swap chain image presentation
		vk::Semaphore presentComplete;
		// Command buffer submission and execution
		vk::Semaphore renderComplete;
	} semaphores;
	std::vector<vk::Fence> fences;
	vk::Pipeline graphicsPipeline;

	vk::DescriptorPool descriptorPool {nullptr};

	void createSyncObjects();

	void destroySyncObjects();

	virtual void createDescriptorPool();
	
	/*
		Create a default renderpass
	*/
	void createRenderPass();
	
	/** @brief prepare frame; submit to queue; presentFrame; updateUniformBuffers*/
	virtual void render() = 0;

	/** @brief create a command pool that supports graphics family */
	void createCommandPool();
	/** @brief create command buffers for swap chains, stored in drawCmdBuffers*/
	void createCommandBuffers();
	/** @brief set up depthStencil member*/
	virtual void createDepthStencil();

	/** @brief (Virtual) Setup default framebuffers for all requested swapchain images */
	virtual void setupFrameBuffer();

	/** begin command buffer, begin renderPass, bind pipeline, set dynamic states */
	virtual void buildCommandBuffer() = 0;

	void recreateSwapChain();

	/* Util functions*/

	void destroyImageData(ImageData& img);

	vk::CommandBuffer beginSingleTimeCommands();

	// Submit to graphics queue and end the buffer
	void endSingleTimeCommands(vk::CommandBuffer commandBuffer);

	/*
	* @brief copy srcBuffer to dstBuffer
	*/
	void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size);

public:
	vki::Instance getInstance();
	/*
	*	Initialize instance and device. Get graphics queue handle as well.
	*/
	virtual void init();

	/** @brief Prepares all Vulkan resources and functions required to run the sample */
	virtual void prepare();

	/** @brief Entry point for the main render loop */
	void renderLoop();

	/** 
	 * Prepare the next frame for workload submission by acquiring the next swap chain image
	 * Should be called at the beginning of user's render()
	 * TODO: Recreate the swapChain if presentation is about to be incompatible
	 */
	void prepareFrame();

	/** 
	 * Presents the current image to the swap chain, recreate if surface resized; then waitIdle
	 * Should be called after queueSubmit in user's render()
	*/
	void presentFrame();

	/** call the render function
	 * TODO: manage frameCounter (fps counter)
	*/
	void nextFrame();

	/*
	* 	Clear all the stuff allocated in init(), prepare(), etc.
	*/
	virtual void clear();

};

//template<typename T>
//vki::Buffer Base::createBufferThroughStaging(const std::vector<T>& data, const vk::BufferUsageFlagBits usage) {
//	vki::Buffer dstBuffer{};
//	vk::DeviceSize bufferSize = sizeof(data[0]) * data.size();
//
//	vki::StagingBuffer staging(device, bufferSize);
//
//	void* mappedData;
//
//	mappedData = device.getDevice().mapMemory(staging.mem, 0, bufferSize);
//	memcpy(mappedData, data.data(), (size_t)bufferSize);
//	device.getDevice().unmapMemory(staging.mem);
//
//	dstBuffer.buffer = device.getDevice().createBuffer(
//		bufferCI.setUsage(usage | vk::BufferUsageFlagBits::eTransferDst)
//	);
//	memReq = device.getDevice().getBufferMemoryRequirements(dstBuffer.buffer);
//	memAI.setAllocationSize(memReq.size);
//	memAI.setMemoryTypeIndex(device.getMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal));
//	device.getDevice().bindBufferMemory(dstBuffer.buffer, dstBuffer.mem);
//
//	copyBuffer(staging.buffer, dstBuffer.buffer, bufferSize);
//
//	staging.clear(device);
//
//	return dstBuffer;
//}