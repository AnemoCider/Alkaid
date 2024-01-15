#pragma once

#include <optional>
#include <iostream>

#include "vulkan/vk_mem_alloc.h"

struct QueueFamilyIndices{
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> transfer;
    std::optional<uint32_t> compute;
};

#define VK_CHECK(x)                                                 \
	do                                                              \
	{                                                               \
		VkResult err = x;                                           \
		if (err)                                                    \
		{                                                           \
			std::cout <<"Vulkan error: " << err << std::endl; \
			abort();                                                \
		}                                                           \
	} while (0)

