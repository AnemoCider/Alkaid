#pragma once

#include "common/VulkanCommon.h"
#include <string>


namespace vki {
	/*
		read a file to string, in binary format
	*/
	std::string readFile(const std::string& filename);
};