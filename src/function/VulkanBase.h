#pragma once

#include "common/VulkanCommon.h"
#include "initialization/VulkanDevice.h"
#include "initialization/VulkanInstance.h"

class Base {

private:

	vki::Instance instance;
	vki::Device device;

public:

	void init();

	void clear();

};