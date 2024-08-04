#pragma once

#include "VulkanContext.h"

namespace vki {

constexpr uint32_t VK_API_VERSION = VK_API_VERSION_1_3;

constexpr uint32_t INVALID_VK_INDEX = 0xFFFFFFFF;

#ifdef NDEBUG
    constexpr bool VK_VALIDATION_ENABLED = false;
#else
    constexpr bool VK_VALIDATION_ENABLED = true;
#endif

class Driver {

public:
    
    static Driver* create();
    ~Driver();

private:

    Driver();

    vk::Instance mInstance = VK_NULL_HANDLE;
    vk::PhysicalDevice mPhysicalDevice = VK_NULL_HANDLE;
    vk::Device mDevice = VK_NULL_HANDLE;
    uint32_t mGraphicsQueueFamilyIndex = INVALID_VK_INDEX;
    uint32_t mGraphicsQueueIndex = INVALID_VK_INDEX;
    vk::Queue mGraphicsQueue = VK_NULL_HANDLE;
    Context mContext = {};
    // VulkanPlatform* mPlatform = nullptr;
    // std::unique_ptr<VulkanTimestamps> mTimestamps;

    // // Placeholder resources
    // VulkanTexture* mEmptyTexture;
    // VulkanBufferObject* mEmptyBufferObject;

    // VulkanSwapChain* mCurrentSwapChain = nullptr;
    // VulkanRenderTarget* mDefaultRenderTarget = nullptr;
    // VulkanRenderPass mCurrentRenderPass = {};
    // VmaAllocator mAllocator = VK_NULL_HANDLE;
    // VkDebugReportCallbackEXT mDebugCallback = VK_NULL_HANDLE;

    // VulkanContext mContext = {};
    // VulkanResourceAllocator mResourceAllocator;
    // VulkanResourceManager mResourceManager;

    // // Used for resources that are created synchronously and used and destroyed on the backend
    // // thread.
    // VulkanThreadSafeResourceManager mThreadSafeResourceManager;

    // VulkanCommands mCommands;
    // VulkanPipelineLayoutCache mPipelineLayoutCache;
    // VulkanPipelineCache mPipelineCache;
    // VulkanStagePool mStagePool;
    // VulkanFboCache mFramebufferCache;
    // VulkanSamplerCache mSamplerCache;
    // VulkanBlitter mBlitter;
    // VulkanSamplerGroup* mSamplerBindings[MAX_SAMPLER_BINDING_COUNT] = {};
    // VulkanReadPixels mReadPixels;
    // VulkanDescriptorSetManager mDescriptorSetManager;

    // VulkanDescriptorSetManager::GetPipelineLayoutFunction mGetPipelineFunction;
};
};