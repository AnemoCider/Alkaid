#pragma once

#include "common/VulkanCommon.h"
#include "initialization/VulkanDevice.h"
#include "initialization/VulkanInstance.h"
#include "preparation/VulkanSwapChain.h"

class Base {

private:

	vki::Instance instance;
	vki::Device device;
	vk::Queue graphicsQueue;
	vki::SwapChain swapChain;
	vk::Pipeline pipeline;

	// Command buffers used for rendering
	std::vector<vk::CommandBuffer> drawCmdBuffers;
	// Global render pass for frame buffer writes
	vk::RenderPass renderPass{ nullptr };
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

public:

	void init();

	void prepare();

	void clear();

	virtual void preparePipeline() = 0;

	virtual void createVertexBuffer() = 0;
	
	/** prepare frame; submit to queue; presentFrame; updateUniformBuffers*/
	virtual void render() = 0;

	/*
		Create a default renderpass
	*/
	void createRenderPass();

	void createCommandBuffers();

	virtual void createDepthStencil();

	/** @brief (Virtual) Setup default framebuffers for all requested swapchain images */
	virtual void setupFrameBuffer();

	/** begin command buffer, begin renderPass, bind pipeline, set dynamic states */
	virtual void buildCommandBuffers() = 0;

	/** @brief Prepares all Vulkan resources and functions required to run the sample */
	virtual void prepare();

	/** @brief Entry point for the main render loop */
	void renderLoop();

	/** Prepare the next frame for workload submission by acquiring the next swap chain image */
	void prepareFrame();
	/** @brief Presents the current image to the swap chain, recreate if surface resized; then waitIdle*/
	void presentFrame();
	/** call the render function, frameCounter++ (fps counter)*/
	void nextFrame();

};