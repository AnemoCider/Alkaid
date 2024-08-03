set(PRIVATE_HEADERS
        
)

# set(PUBLIC_HEADERS
#         include/Engine.h
#         include/Application.h
#         include/VertexBuffer.h
# )

set(BACKEND_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/backend/src)

set(BACKEND_SRCS
        ${BACKEND_SRC_DIR}/vulkan/VulkanAsset.cpp
        # src/vulkan/VulkanBase.cpp
        ${BACKEND_SRC_DIR}/vulkan/VulkanBuffer.cpp
        ${BACKEND_SRC_DIR}/vulkan/VulkanCommand.cpp
        ${BACKEND_SRC_DIR}/vulkan/VulkanCamera.cpp
        ${BACKEND_SRC_DIR}/vulkan/VulkanDevice.cpp
        ${BACKEND_SRC_DIR}/vulkan/VulkanInstance.cpp
        ${BACKEND_SRC_DIR}/vulkan/VulkanPipeline.cpp
        ${BACKEND_SRC_DIR}/vulkan/VulkanSwapChain.cpp
        ${BACKEND_SRC_DIR}/vulkan/VulkanUtils.cpp
)

set(ENGINE_SRCS
        src/Engine.cpp
        src/Application.cpp
        src/VertexBuffer.cpp
)

# set(KTX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../external/ktx)
# set(KTX_SOURCES
#     ${KTX_DIR}/lib/texture.c
#     ${KTX_DIR}/lib/hashlist.c
#     ${KTX_DIR}/lib/checkheader.c
#     ${KTX_DIR}/lib/swap.c
#     ${KTX_DIR}/lib/memstream.c
#     ${KTX_DIR}/lib/filestream.c )

# set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../external/imgui)
# file(GLOB IMGUI_SOURCES
#     "${IMGUI_DIR}/*.cpp"
#     "${IMGUI_DIR}/*.h"
#     ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
#     ${IMGUI_DIR}/backends/imgui_impl_glfw.h
#     ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
#     ${IMGUI_DIR}/backends/imgui_impl_vulkan.h
# )

add_library(base STATIC ${BACKEND_SRCS} ${ENGINE_SRCS} ${PUBLIC_HEADERS} ${PRIVATE_HEADERS}) # ${KTX_SOURCES} ${IMGUI_SOURCES})
# source_group("Header Files" FILES ${PUBLIC_HEADERS} ${PRIVATE_HEADERS})
source_group("Backend sources" FILES ${BACKEND_SRCS})
source_group("Engine sources" FILES ${ENGINE_SRCS})
# source_group("KTX sources" FILES ${KTX_SOURCES})
# source_group("ImGUI sources" FILES ${IMGUI_SOURCES})
target_link_libraries(base PUBLIC ${Vulkan_LIBRARY})
target_include_directories(base PRIVATE ${BACKEND_SRC_DIR})
target_include_directories(base PUBLIC include)