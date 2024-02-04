#pragma once

#include "common/VulkanCommon.h"
#include "asset/VulkanAsset.h"
#include "initialization/VulkanDevice.h"
#include <string>

namespace vki {
	class Pipeline {
	private:
		std::string shaderPath = "";
		std::string shaderName = "";
		vki::Device* device;

		vk::ShaderModule vertexShader;
		vk::ShaderModule fragmentShader;

		vk::Pipeline pipeline;

		vk::PipelineShaderStageCreateInfo shaderStageCI{};
		vk::PipelineDynamicStateCreateInfo dynamicCI{};
		vk::PipelineVertexInputStateCreateInfo vertexInputCI{};
		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyCI{};
		vk::Viewport viewport{};
		vk::PipelineViewportStateCreateInfo viewportCI{};
		vk::Rect2D scissor{};
		vk::PipelineRasterizationStateCreateInfo rasterizerCI{};
		vk::PipelineMultisampleStateCreateInfo multisamplingCI{};
		vk::PipelineColorBlendStateCreateInfo colorBlendingCI{};
		vk::PipelineLayoutCreateInfo pipelineLayoutCI{};

	public:

		void setDevice(vki::Device* device);
		void setShaderPath(const std::string& path);
		void setShaderName(const std::string& name);
		virtual std::string readShader(const std::string& suffix = ".vert");
		void setVertShader(const std::string& code);
		void setFragShader(const std::string& code);
		void init();

	};
	
	
};