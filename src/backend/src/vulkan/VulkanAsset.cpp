#include "VulkanAsset.h"
#include <fstream>
#include <sstream>  

std::string vki::readFile(const std::string& filename) {
	std::ifstream file(filename, std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("failed to open file!");
	}

	std::stringstream buffer{};
	buffer << file.rdbuf();
	file.close();
	return buffer.str();
}
