# Design Draft

## High Level Abstraction

## Low Level Abstraction

### Data preparation

### Core Rendering Process

- Acquire next image
- Begin command buffer
- Begin render pass
- Set states: viewport, scissor
- Bind pipeline
- Bind descriptor sets
- Bind vertex & index buffers
- Draw
- End render pass
- End command buffer
- Queue present

## Program Initialization

### Create Instance

Requires:

- Vulkan version
- instanceExtensions
- layer

Produces:

- instance

### Pick Physical Device

Requires:

null

Produces:

- Formats

### Create Logical Device

Requires:

- Features to enable
- Device Extensions to enable
- Queue families to enable

Produces:

- handle to the queue

### Create Window and surface

Requires:

- Dimensions of the window

Produces:

- window
- Registered class pointer
- surface

## Preparation

### Swap Chain

Requires:

- surface

Produces:

- swap chain images
- swap chain image views
- Physical Device Properties
  - deviceName
  - limits
- Physical Device Features
- Physical Device Memory Properties

### Command Pool

Requires:

- queue family index

### Command Buffer

One for each frame

### Synchronization Objects

### Depth Stencil

Requires:

- format, from physical device
- mipmap levels
- sampling count

Provides:

- depthStencilImageView

### Frame buffer

Very flexible

## Assets

- loadShader
- loadObjModel

## Control

- camera class