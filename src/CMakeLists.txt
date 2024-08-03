set(BACKEND_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/backend/src)
set(ENGINE_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

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
        ${ENGINE_SRC_DIR}/Engine.cpp
)

set(ENGINE_MODULE_INTERFACES
        ${ENGINE_SRC_DIR}/Engine.cppm
        ${ENGINE_SRC_DIR}/Application.cppm
        ${ENGINE_SRC_DIR}/VertexBuffer.cppm
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

add_library(base STATIC) #  ${PUBLIC_HEADERS} ${PRIVATE_HEADERS} ${KTX_SOURCES} ${IMGUI_SOURCES})

target_sources(base
    PUBLIC
        FILE_SET CXX_MODULES FILES
        ${ENGINE_MODULE_INTERFACES})

target_sources(base
    PRIVATE
        ${BACKEND_SRCS} ${ENGINE_SRCS}
)

# source_group("Header Files" FILES ${PUBLIC_HEADERS} ${PRIVATE_HEADERS})
source_group("Engine interfaces" FILES ${ENGINE_MODULE_INTERFACES})
source_group("Backend sources" FILES ${BACKEND_SRCS})
source_group("Engine sources" FILES ${ENGINE_SRCS})
# source_group("KTX sources" FILES ${KTX_SOURCES})
# source_group("ImGUI sources" FILES ${IMGUI_SOURCES})
target_link_libraries(base PUBLIC ${Vulkan_LIBRARY})
target_include_directories(base PRIVATE ${BACKEND_SRC_DIR})
target_include_directories(base PUBLIC include)

# install(
#         DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/base.dir/
#         DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/ifc
#         FILES_MATCHING PATTERN "*.ifc"
#     )

# Create a list of .ifc files to be copied
file(GLOB IFC_FILES "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/base.dir/*.ifc")

# Copy .ifc files for, e.g., intellisense usage
foreach(ifc_file IN LISTS IFC_FILES)
    get_filename_component(ifc_filename ${ifc_file} NAME)
    add_custom_command(
        TARGET base POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${ifc_file}
                ${CMAKE_BINARY_DIR}/ifc/${ifc_filename}
        COMMENT "Copying ${ifc_filename} after build."
    )
endforeach()