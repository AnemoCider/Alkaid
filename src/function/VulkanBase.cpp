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


void Base::prepare() {
	swapChain.setDevice(&device);
	swapChain.setInstance(&instance);
	swapChain.init();

	preparePipeline();
}


void Base::clear() {
	device.getDevice().destroyPipeline(pipeline);
	swapChain.clear();
	device.clear();
	instance.destroyWindow();
	instance.clear();
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
		.colorAttachmentCount = 1,
		.pColorAttachments = &colorRef,
		.pDepthStencilAttachment = &depthRef,
		.inputAttachmentCount = 0,
		.pInputAttachments = nullptr,
		.preserveAttachmentCount = 0,
		.pPreserveAttachments = nullptr,
		.pResolveAttachments = nullptr
	};

	std::array<vk::SubpassDependency, 2> dependencies;
	// depth attachment dependencies
	// Essentially 2 things:
	// depth write happens after previous frame read
	// depth read happens after previous frame write
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
}