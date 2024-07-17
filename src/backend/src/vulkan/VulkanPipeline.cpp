#include "VulkanPipeline.h"
#include "VulkanAsset.h"

void vki::Pipeline::setDevice(vki::Device* device) {
	this->device = device;
}

void vki::Pipeline::setShaderPath(const std::string& path) {
	shaderPath = path;
}

void vki::Pipeline::setShaderName(const std::string& name) {
	shaderName = name;
}

std::string vki::Pipeline::readShader(const std::string& suffix) {
	return vki::readFile(shaderPath + shaderName + suffix);
}

vk::ShaderModule vki::Pipeline::createVertShader(const std::string& code) {
	vk::ShaderModuleCreateInfo shaderCI{
		.codeSize = code.size(),
		.pCode = reinterpret_cast<const uint32_t*>(code.data())
	};

	return device->getDevice().createShaderModule(shaderCI, nullptr);
}

vk::ShaderModule vki::Pipeline::createFragShader(const std::string& code) {
	vk::ShaderModuleCreateInfo shaderCI{
		.codeSize = code.size(),
		.pCode = reinterpret_cast<const uint32_t*>(code.data())
	};

	return device->getDevice().createShaderModule(shaderCI, nullptr);
}

void vki::Pipeline::initCreateInfo()
{
	vertexInputCI = {

	};
	pipelineInfo = {
		.pVertexInputState = &vertexInputCI,
		.pInputAssemblyState = &inputAssemblyCI,
		.pViewportState = &viewportCI,
		.pRasterizationState = &rasterizerCI,
		.pMultisampleState = &multisamplingCI,
		.pDepthStencilState = &depthStateCI,
		.pColorBlendState = &colorBlendingCI,
		.pDynamicState = &dynamicCI,
	};
	
}

vk::GraphicsPipelineCreateInfo& vki::Pipeline::getPipelineCI() {
	return pipelineInfo;
}

//void vki::Pipeline::init() {
//	pipeline = device->getDevice().createGraphicsPipeline(nullptr, pipelineInfo).value;
//}
//
//void vki::Pipeline::clear() {
//	device->getDevice().destroyPipeline(pipeline);
//}

void vki::Pipeline::clearShaderModule(vk::ShaderModule& module) {
	device->getDevice().destroyShaderModule();
}