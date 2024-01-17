# Design Draft

## base

### Usage

The following functions may be overriden:

- isPhysicalDeviceSuitable: the first suitable device will be selected as the physical device.
- createDepthStencil
- createFrameBuffers: By default use 2 attachments for each framebuffer
- createRenderPass: By default, use the first attachment as color attachment, and the second as depth buffer.

The following must be overriden (as they are declared pure virtual):

- getShaderPathName: return the path of shader, plus the name without suffix, relative to the main cpp file.
- createCommandBuffers: because it relies on the maxFrameCount, which is defined as a static constant in each example.
- render: called in renderLoop.

### References

Vulkan Guide:

```Cpp
void init_vulkan();
void init_swapchain();
void init_default_renderpass();
void init_framebuffers();
void init_commands();
void init_sync_structures();
void init_pipelines();
void init_scene();
void init_descriptors();
```

Vulkan Examples:

```Cpp
uint32_t destWidth;
uint32_t destHeight;
bool resizing = false;
void nextFrame();
void createPipelineCache();
void createCommandPool();
void createSynchronizationPrimitives();
void initSwapchain();
void setupSwapChain();
void createCommandBuffers();
void destroyCommandBuffers();

virtual VkResult createInstance(bool enableValidation);
virtual void render() = 0;
virtual void viewChanged();
/** @brief (Virtual) Called when the window has been resized, can be used by the sample application to recreate resources */
virtual void windowResized();
/** @brief (Virtual) Called when resources have been recreated that require a rebuild of the command buffers (e.g. frame buffer), to be implemented by the sample application */
virtual void buildCommandBuffers();
/** @brief (Virtual) Setup default depth and stencil views */
virtual void setupDepthStencil();
/** @brief (Virtual) Setup default framebuffers for all requested swapchain images */
virtual void setupFrameBuffer();
/** @brief (Virtual) Setup a default renderpass */
virtual void setupRenderPass();
/** @brief (Virtual) Called after the physical device features have been read, can be used to set features to enable on the device */
virtual void getEnabledFeatures();
/** @brief (Virtual) Called after the physical device extensions have been read, can be used to enable extensions based on the supported extension listing*/
virtual void getEnabledExtensions();
/** @brief Prepares all Vulkan resources and functions required to run the sample */
virtual void prepare();
/** @brief Loads a SPIR-V shader file for the given shader stage */
VkPipelineShaderStageCreateInfo loadShader(std::string fileName, VkShaderStageFlagBits stage);
void windowResize();
/** @brief Entry point for the main render loop */
void renderLoop();
/** Prepare the next frame for workload submission by acquiring the next swap chain image */
void prepareFrame();
/** @brief Presents the current image to the swap chain */
void submitFrame();
/** @brief (Virtual) Default image acquire + submission and command buffer submission function */
virtual void renderFrame();
```

## Device

Logical device related stuff

Input:

- Physical Device
- Enabled Device Features
- Enabled Layers
- Enabled Extensions
- Queue families to use

Interfaces:

- Check device extension support
- Get queue family indices
- Create logical device

## Swapchain

Input:

- Instance (to manage surface)
- Physical device
- Logical device
- Surface

Interfaces:

- Create swap chain
- Cleanup (surface, image views, and swap chain)
