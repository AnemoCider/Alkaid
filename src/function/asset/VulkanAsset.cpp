#include "VulkanAsset.h"

std::string vki::readFile(const std::string& filename) {
	std::ifstream file(filename, std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("failed to open file!");
	}

	std::stringstream buffer;
	buffer << file.rdbuf();
	return buffer.str();
}
