set(PUBLIC_HEADER_DIR include)

set(PRIVATE_HEADERS
        
)

set(PUBLIC_HEADERS
        
)

set(SRCS
        src/vulkan/VulkanAsset.cpp
        src/vulkan/VulkanBase.cpp
        src/vulkan/VulkanBuffer.cpp
        src/vulkan/VulkanCommand.cpp
        src/vulkan/VulkanCamera.cpp
        src/vulkan/VulkanDevice.cpp
        src/vulkan/VulkanInstance.cpp
        src/vulkan/VulkanPipeline.cpp
        src/vulkan/VulkanSwapChain.cpp
        src/vulkan/VulkanUtils.cpp
)

set(KTX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../external/ktx)
set(KTX_SOURCES
    ${KTX_DIR}/lib/texture.c
    ${KTX_DIR}/lib/hashlist.c
    ${KTX_DIR}/lib/checkheader.c
    ${KTX_DIR}/lib/swap.c
    ${KTX_DIR}/lib/memstream.c
    ${KTX_DIR}/lib/filestream.c )

set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../external/imgui)
file(GLOB IMGUI_SOURCES
    "${IMGUI_DIR}/*.cpp"
    "${IMGUI_DIR}/*.h"
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.h
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.h
)

add_library(base STATIC ${BASE_SRC} ${FUNC_SRC} ${PUBLIC_HEADERS} ${PRIVATE_HEADERS} ${KTX_SOURCES} ${IMGUI_SOURCES})
source_group("Header Files" FILES ${PUBLIC_HEADERS} ${PRIVATE_HEADERS})
source_group("KTX sources" FILES ${KTX_SOURCES})
source_group("ImGUI sources" FILES ${IMGUI_SOURCES})
target_link_libraries(base PUBLIC ${Vulkan_LIBRARY})
target_include_directories(base PUBLIC ${PUBLIC_HEADER_DIR})