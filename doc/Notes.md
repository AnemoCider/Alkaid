# Notes during development

## RenderPass

### Transition and Dependency

#### Question

How to understand this:

```Cpp
VkAttachmentReference attachmentReference = {
    .attachment = 0,
    .layout     = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL};

// Subpass containing first draw
VkSubpassDescription subpass = {
    ...
    .colorAttachmentCount = 1,
    .pColorAttachments = &attachmentReference,
    ...};

/* Add external dependencies to ensure that the layout
   transitions happen at the right time.
   Signal operations happen
   at COLOR_ATTACHMENT_OUTPUT to reduce their scope to
   the minimum; the subpass dependencies are then both
   adjusted to match */
VkSubpassDependency dependencies[2] = {
    {
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = VK_ACCESS_NONE_KHR,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dependencyFlags = 0
    }, ...
}
```

Traditionally, the pipeline's color attachment output stage is blocked until vkAcquireNextImage signals the imageAvailable semaphore. However, the image transition and clear by loadOp and initialLayout, by default, happens at the beginning of the renderpass, and may happen before the image is acquired. Therefore, we have two options to synchronize it:

#### Block the start of renderpass until imageAvailable

First is to change the waitStages from color attachment out to start of the pipeline.

The second option is to use subpass dependencies.

All operations submitted in the current queueSubmit will be blocked on dstWaitStages, which is VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT in this example, until the semaphore is signaled.

> The second synchronization scope includes every command submitted in the same batch. In the case of vkQueueSubmit, the second synchronization scope is limited to operations on the pipeline stages determined by the destination stage mask specified by the corresponding element of pWaitDstStageMask. Also, in the case of vkQueueSubmit, the second synchronization scope additionally includes all commands that occur later in submission order.

to be simple,

> if you wait on a semaphore, all commands in the VkSubmitInfo can reach a pWaitDstStageMask stage only after the semaphore was signaled

Then, the layout transition is chained after it:

> Automatic layout transitions away from initialLayout happens-after the availability operations for all dependencies with a srcSubpass equal to VK_SUBPASS_EXTERNAL.

Availability operation:

> The availability operation (if any) happens-after the source synchronization scope. Then happens the layout transition (if any). Then happens the visibility op (if any). And only after then can the destination synchronization scope execute.

Therefore, the transition happens right after the latest source synchronization scope among all dependencies with srcSubpass equal to VK_SUBPASS_EXTERNAL.

Therefore, to postpone the transition until after the image is available, we just set the dependency's source synchronization scope to be the time the semaphore is signaled, i.e., VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT.

#### What about loadOp

> The load operation for each sample in an attachment happens-before any recorded command which accesses the sample in the first subpass where the attachment is used. [...] Load operations for attachments with a color format execute in the VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT pipeline stage.
>
> VK_ATTACHMENT_LOAD_OP_LOAD [...] For attachments with a color format, this uses the access type VK_ACCESS_COLOR_ATTACHMENT_READ_BIT.
>
> VK_ATTACHMENT_LOAD_OP_CLEAR(or VK_ATTACHMENT_LOAD_OP_DONT_CARE) [...] For attachments with a color format, this uses the access type VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT.

Therefore, LOAD_OP_CLEAR is blocked by the semaphore. And when the semaphore is signaled, automatic layout transition takes place immediately. Then happens the visibility op, followed by loadOp.