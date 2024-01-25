#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"

#include <vulkan/vk_mem_alloc.h>
#include "VulkanBase.h"

#include <ktx.h>
#include <ktxvulkan.h>

#include <memory>
#include <iostream>
#include <array>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <chrono>

#include "VulkanInit.h"

using namespace tinygltf;

class PCSS : public VulkanBase {

	Model model;
	TinyGLTF loader;
	
	void getModelPath() {}

	void loadModel() {}
};

int main() {
	std::cout << "test\n";
}