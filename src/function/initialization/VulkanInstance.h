#pragma once

#include "common/VulkanCommon.h"

#include <vector>
#include <string>

namespace vki {

	class Instance {

	private:

		bool enableValidationLayer = true;
		bool isPhyDeviceSuitable(const vk::PhysicalDevice& device);

	public:


		vk::Instance instance;
		std::vector<const char*> extensions;
		std::vector<const char*> phyExtensions;
		std::vector<const char*> layers;

		vk::PhysicalDevice phyDevice;

		GLFWwindow* window;
		uint32_t width = 800;
		uint32_t height = 600;
		vk::SurfaceKHR surface;

		uint32_t grqFamilyIndex;

		Instance() = default;
		~Instance() = default;

		/*
			Initializes vulkan instance.
			Call this after setting all desired states.
		*/
		void init();

		/*
			Add new instanceextensions.
			By default, extensions required by glfw are added.
			@param a vector containing names of extensions to add.
		*/
		void addExtensions(const std::vector<const char*>& ext);

		/*
			Add new layers.
			By default, added validation layer.
			@param a vector containing names of layers to add.
		*/
		void addLayers(const std::vector<const char*>& layers);

		/*
			Destroy the instance.
		*/
		void clear();

		void setValidationLayer(bool enable);

		void pickPhysicalDevice();

		void getGraphicsQueue();

		/*
			Create glfw window and surface
		*/
		void createWindow(void* ptr);

		void destroyWindow();

	};

};