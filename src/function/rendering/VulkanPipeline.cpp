#include "VulkanPipeline.h"

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
	return readFile(shaderPath + shaderName + suffix);
}

void vki::Pipeline::setVertShader(const std::string& code) {
	vk::ShaderModuleCreateInfo shaderCI{
		.codeSize = code.size(),
		.pCode = reinterpret_cast<const uint32_t*>(code.data())
	};

	vertexShader = device->getDevice().createShaderModule(shaderCI, nullptr);
}

void vki::Pipeline::setFragShader(const std::string& code) {
	vk::ShaderModuleCreateInfo shaderCI{
		.codeSize = code.size(),
		.pCode = reinterpret_cast<const uint32_t*>(code.data())
	};

	fragmentShader = device->getDevice().createShaderModule(shaderCI, nullptr);
}

