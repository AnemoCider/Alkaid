#pragma once

#include "common/VulkanCommon.h"

namespace vki {

	uint32_t getMemoryTypeIndex(uint32_t typeBits, vk::MemoryPropertyFlagBits properties) {
		// Iterate over all memory types available for the device used in this example
		for (uint32_t i = 0; i < deviceMemoryProperties.memoryTypeCount; i++) {
			if ((typeBits & 1) == 1) {
				if ((deviceMemoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
					return i;
				}
			}
			typeBits >>= 1;
		}

		throw "Could not find a suitable memory type!";
	}

};