#pragma once

#include "VulkanCommon.h"
#include "VulkanAsset.h"
#include "VulkanDevice.h"
#include <string>

namespace vki {
	class Pipeline {
	private:
		std::string shaderPath = "";
		std::string shaderName = "";
		vki::Device* device;

		vk::ShaderModule vertexShader;
		vk::ShaderModule fragmentShader;

		vk::GraphicsPipelineCreateInfo pipelineInfo;
		vk::Pipeline pipeline;

		vk::PipelineDynamicStateCreateInfo dynamicCI{};
		vk::PipelineVertexInputStateCreateInfo vertexInputCI{};
		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyCI{};
		vk::Viewport viewport{};
		vk::PipelineViewportStateCreateInfo viewportCI{};
		vk::Rect2D scissor{};
		vk::PipelineRasterizationStateCreateInfo rasterizerCI{};
		vk::PipelineMultisampleStateCreateInfo multisamplingCI{};
		vk::PipelineDepthStencilStateCreateInfo depthStateCI{};
		vk::PipelineColorBlendStateCreateInfo colorBlendingCI{};

		vk::PipelineLayout pipelineLayout{};
		vk::RenderPass renderPass;

		void initCreateInfo();

	public:

		void setDevice(vki::Device* device);
		void setShaderPath(const std::string& path);
		void setShaderName(const std::string& name);
		std::string readShader(const std::string& suffix = ".vert");
		vk::GraphicsPipelineCreateInfo& getPipelineCI();
		vk::ShaderModule createVertShader(const std::string& code);
		vk::ShaderModule createFragShader(const std::string& code);
		/*
			Initialize the pipeline.
			Note that many key properties are not set, e.g., 
				shaderStages, renderPass, and pipelineLayout.
			Set these through getPipelineCI, before calling this function.
			TODO: implement pipeline cache to make pipeline recreation faster.
		*/
		/*void init();
		void clear();*/

		/**/
		void clearShaderModule(vk::ShaderModule& module);
	};
	
	
};