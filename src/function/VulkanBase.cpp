#include "VulkanBase.h"
#include <vector>

void Base::init() {
	instance.init();
	instance.createWindow(this);

	device.setInstance(&instance);
#ifdef __APPLE__
	device.addExtension("VK_KHR_portability_subset");
#endif
	device.init();
	device.getGraphicsQueue(graphicsQueue);
}

void Base::setupFrameBuffer() {
	std::vector<vk::ImageView> attachments(2);
	frameBuffers.resize(swapChain.getImageCount());
	vk::FramebufferCreateInfo frameBufferCI {
		.renderPass = renderPass,
		.attachmentCount = static_cast<uint32_t>(attachments.size()),
		.pAttachments = attachments.data(),
		.width = instance.width,
		.height = instance.height,
		.layers = 1
	};

	for (uint32_t i = 0; i < frameBuffers.size(); i++) {
		attachments[0] = swapChain.views[i];
		attachments[1] = depthStencil.view;
		frameBuffers[i] = device.getDevice().createFramebuffer(frameBufferCI, nullptr);
	}
}

void Base::prepare()
{
    swapChain.setDevice(&device);
	swapChain.setInstance(&instance);
	swapChain.init();

	createRenderPass();
	createDepthStencil();
	setupFrameBuffer();
	createCommandPool();
	createCommandBuffers();
	createSyncObjects();
	createDescriptorPool();
}

void Base::destroyImageData(ImageData& img) {
	device.getDevice().destroyImageView(img.view);
	device.getDevice().freeMemory(img.mem);
	device.getDevice().destroyImage(img.image);
}

vk::CommandBuffer Base::beginSingleTimeCommands() {
	vk::CommandBufferAllocateInfo allocInfo{
		.commandPool = commandPool,
		.level = vk::CommandBufferLevel::ePrimary,
		.commandBufferCount = 1
	};

	vk::CommandBuffer commandBuffer;
	commandBuffer = device.getDevice().allocateCommandBuffers(allocInfo).front();

	vk::CommandBufferBeginInfo beginInfo{
		.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
	};

	commandBuffer.begin(beginInfo);

	return commandBuffer;
}

void Base::endSingleTimeCommands(vk::CommandBuffer commandBuffer) {
	commandBuffer.end();

	vk::SubmitInfo submitInfo{
		.commandBufferCount = 1,
		.pCommandBuffers = &commandBuffer
	};

	graphicsQueue.submit(submitInfo);

	graphicsQueue.waitIdle();

	device.getDevice().freeCommandBuffers(commandPool, commandBuffer);
}

void Base::copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) {
	auto commandBuffer = beginSingleTimeCommands();

	vk::BufferCopy copyRegion{
		.size = size
	};

	commandBuffer.copyBuffer(srcBuffer, dstBuffer, copyRegion);

	endSingleTimeCommands(commandBuffer);
}

void Base::renderLoop() {
	while (!glfwWindowShouldClose(instance.window)) {
		nextFrame();
	}
}

void Base::prepareFrame() {
	auto result = device.getDevice().acquireNextImageKHR(swapChain.getSwapChain(), UINT64_MAX,
		semaphores.presentComplete, nullptr, &currentBuffer);
	// Incompatible
	if (result == vk::Result::eErrorOutOfDateKHR) {
		// TODO: recreate the swap chain
	} else {
		// Compatible, but may not exactly match
		assert (result == vk::Result::eSuccess || result == vk::Result::eSuboptimalKHR);
	}
	
}

void Base::presentFrame() {
	vk::PresentInfoKHR presentInfo{
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &semaphores.renderComplete,
		.swapchainCount = 1,
		.pSwapchains = &swapChain.getSwapChain(),
		.pImageIndices = &currentBuffer
	};
	graphicsQueue.presentKHR(presentInfo);
}

void Base::nextFrame() {
	render();
}


void Base::clear() {
	device.getDevice().destroyDescriptorPool(descriptorPool);
	destroySyncObjects();
	device.getDevice().destroyCommandPool(commandPool);
	destroyImageData(depthStencil);
	for (uint32_t i = 0; i < frameBuffers.size(); i++) {
		device.getDevice().destroyFramebuffer(frameBuffers[i]);
	}
	device.getDevice().destroyPipeline(pipeline);
	swapChain.clear();
	device.clear();
	instance.destroyWindow();
	instance.clear();
}

void Base::createSyncObjects() {
	vk::FenceCreateInfo fenceCI {};
	fences.resize(drawCmdBuffers.size());
	for (auto& f : fences) {
		f = device.getDevice().createFence(fenceCI);
	}
	vk::SemaphoreCreateInfo semaphoreCI{};
	semaphores.presentComplete = device.getDevice().createSemaphore(semaphoreCI);
	semaphores.renderComplete = device.getDevice().createSemaphore(semaphoreCI);
}

void Base::destroySyncObjects() {
	device.getDevice().destroySemaphore(semaphores.renderComplete);
	device.getDevice().destroySemaphore(semaphores.presentComplete);
	for (auto& f : fences) {
		device.getDevice().destroyFence(f);
	}
}

void Base::createDescriptorPool() {
	std::vector<vk::DescriptorPoolSize> poolSizes(2);
	poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
	poolSizes[0].descriptorCount = 1;
	poolSizes[1].type = vk::DescriptorType::eCombinedImageSampler;
	poolSizes[1].descriptorCount = 1;

	vk::DescriptorPoolCreateInfo poolInfo {
		.maxSets = static_cast<uint32_t>(drawCmdBuffers.size()),
		.poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
		.pPoolSizes = poolSizes.data()
	};
	descriptorPool = device.getDevice().createDescriptorPool(poolInfo);
}

void Base::createRenderPass() {
	vk::AttachmentDescription colorDescription {
		.format = swapChain.getColorFormat(),
		.samples = vk::SampleCountFlagBits::e1,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eStore,
		.stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
		.stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
		.initialLayout = vk::ImageLayout::eUndefined,
		.finalLayout = vk::ImageLayout::ePresentSrcKHR
	};
	vk::AttachmentDescription depthDescription {
		.format = instance.depthFormat,
		.samples = vk::SampleCountFlagBits::e1,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eDontCare,
		.stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
		.stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
		.initialLayout = vk::ImageLayout::eUndefined,
		.finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal
	};
	vk::AttachmentReference colorRef {
		.attachment = 0,
		.layout = vk::ImageLayout::eColorAttachmentOptimal
	};
	vk::AttachmentReference depthRef {
		.attachment = 1,
		.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal
	};
	std::array<vk::AttachmentDescription, 2> descriptions {
		colorDescription, depthDescription
	};
	vk::SubpassDescription subPass {
		.pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
		.inputAttachmentCount = 0,
		.pInputAttachments = nullptr,
		.colorAttachmentCount = 1,
		.pColorAttachments = &colorRef,
		.pResolveAttachments = nullptr,
		.pDepthStencilAttachment = &depthRef,
		.preserveAttachmentCount = 0,
		.pPreserveAttachments = nullptr,
	};

	std::array<vk::SubpassDependency, 2> dependencies;
	// depth attachment dependencies
	// postpone loadOp after previous renderPass write
	dependencies[0] = {
		.srcSubpass = vk::SubpassExternal,
		.dstSubpass = 0,
		.srcStageMask = 
			vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests,
		.dstStageMask = 
			vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests,
		.srcAccessMask = 
			vk::AccessFlagBits::eDepthStencilAttachmentWrite,
		.dstAccessMask = 
			vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite,
	};
	dependencies[1] = {
		.srcSubpass = vk::SubpassExternal,
		.dstSubpass = 0,
		.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
		.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
		.srcAccessMask = vk::AccessFlagBits::eNone,
		.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eColorAttachmentRead
	};

	vk::RenderPassCreateInfo renderPassCI {
		.attachmentCount = static_cast<uint32_t>(descriptions.size()),
		.pAttachments = descriptions.data(),
		.subpassCount = 1,
		.pSubpasses = &subPass,
		.dependencyCount = static_cast<uint32_t>(dependencies.size()),
		.pDependencies = dependencies.data() 
	};
	
	renderPass = device.getDevice().createRenderPass(renderPassCI, nullptr);
}

void Base::createCommandPool() {
	vk::CommandPoolCreateInfo poolCI {
		.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
		.queueFamilyIndex = instance.grqFamilyIndex
	};
	commandPool = device.getDevice().createCommandPool(poolCI);
}

void Base::createCommandBuffers() {
	drawCmdBuffers.resize(swapChain.getImageCount());
	vk::CommandBufferAllocateInfo cmdBufferAI {
		.commandPool = commandPool,
		.level = vk::CommandBufferLevel::ePrimary,
		.commandBufferCount = static_cast<uint32_t>(drawCmdBuffers.size())
	};
	drawCmdBuffers = device.getDevice().allocateCommandBuffers(cmdBufferAI);
}

void Base::createDepthStencil() {
	vk::ImageCreateInfo imageCI {
		.imageType = vk::ImageType::e2D,
		.format = instance.depthFormat,
		.extent = {instance.width, instance.height, 1},
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = vk::SampleCountFlagBits::e1,
		.tiling = vk::ImageTiling::eOptimal,
		.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment
	};
	depthStencil.image = device.getDevice().createImage(imageCI, nullptr);

	auto memReq = device.getDevice().
		getImageMemoryRequirements(depthStencil.image);
	
	vk::MemoryAllocateInfo memAI {
		.allocationSize = memReq.size,
		.memoryTypeIndex = device.getMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)
	};
	depthStencil.mem = device.getDevice().allocateMemory(memAI);
	device.getDevice().bindImageMemory(depthStencil.image, depthStencil.mem, 0);

	vk::ImageSubresourceRange range{
		.aspectMask = vk::ImageAspectFlagBits::eDepth,
		.baseMipLevel = 0,
		.levelCount = 1,
		.baseArrayLayer = 0,
		.layerCount = 1,
	};

	vk::ImageViewCreateInfo imageViewCI{
		.image = depthStencil.image,
		.viewType = vk::ImageViewType::e2D,
		.format = instance.depthFormat,
		.subresourceRange = range
	};
	// Stencil aspect should only be set on depth + stencil formats (VK_FORMAT_D16_UNORM_S8_UINT..VK_FORMAT_D32_SFLOAT_S8_UINT
	if (instance.depthFormat >= vk::Format::eD16UnormS8Uint) {
		imageViewCI.subresourceRange.aspectMask |= vk::ImageAspectFlagBits::eStencil;
	}
	depthStencil.view = device.getDevice().createImageView(imageViewCI);
}
