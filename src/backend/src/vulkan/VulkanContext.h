#pragma once

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>

#include <vector>

namespace vki {

struct Context {
    vk::PhysicalDeviceProperties mPhysicalDeviceProperties;
    vk::PhysicalDeviceFeatures mPhysicalDeviceFeatures;
    vk::PhysicalDeviceMemoryProperties mPhysicalDeviceMemoryProperties;
};

}