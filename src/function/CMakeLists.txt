file(GLOB_RECURSE CUR_FUNC_SRC "*.cpp")
set(FUNC_SRC ${CUR_FUNC_SRC} PARENT_SCOPE)

set(KTX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../external/ktx)
set(KTX_SOURCES
    ${KTX_DIR}/lib/texture.c
    ${KTX_DIR}/lib/hashlist.c
    ${KTX_DIR}/lib/checkheader.c
    ${KTX_DIR}/lib/swap.c
    ${KTX_DIR}/lib/memstream.c
    ${KTX_DIR}/lib/filestream.c )

add_library(base STATIC ${BASE_SRC} ${KTX_SOURCES})
target_link_libraries(base ${Vulkan_LIBRARY})