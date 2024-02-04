#pragma once

#include "common/VulkanCommon.h"
#include <string>
#include <fstream>

namespace vki {
	/*
		read a file to string, in binary format
	*/
	inline static std::string readFile(const std::string& filename);
};