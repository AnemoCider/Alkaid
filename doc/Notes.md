# Notes during development

## RenderPass

### Transition and Dependency

Traditionally, the pipeline's color attachment output stage is blocked until vkAcquireNextImage signals the imageAvailable semaphore. However, the image transition and clear by loadOp and initialLayout happens at the beginning of the renderpass, and may happen before the image is acquired.

#### Block the start of renderpass until imageAvailable

Change the waitStages from color attachment out to start of the pipeline, or

The first subpass should not happen until all previous color attachment output operations (including acquireNextImage) have finished.

> If srcSubpass is equal to VK_SUBPASS_EXTERNAL, the first synchronization scope includes commands that occur earlier in submission order than the vkCmdBeginRenderPass used to begin the render pass instance. 