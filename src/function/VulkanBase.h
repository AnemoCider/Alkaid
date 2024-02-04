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

public:

	void init();

	void prepare();

	void clear();

};