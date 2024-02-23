file(GLOB_RECURSE FUNC_SRC "*.cpp")
file(GLOB_RECURSE HEADERS "*.h")

set(KTX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../external/ktx)
set(KTX_SOURCES
    ${KTX_DIR}/lib/texture.c
    ${KTX_DIR}/lib/hashlist.c
    ${KTX_DIR}/lib/checkheader.c
    ${KTX_DIR}/lib/swap.c
    ${KTX_DIR}/lib/memstream.c
    ${KTX_DIR}/lib/filestream.c )

add_library(base STATIC ${BASE_SRC} ${FUNC_SRC} ${HEADERS} ${KTX_SOURCES})
source_group("Header Files" FILES ${HEADERS})
source_group("KTX sources" FILES ${KTX_SOURCES})
target_link_libraries(base ${Vulkan_LIBRARY})