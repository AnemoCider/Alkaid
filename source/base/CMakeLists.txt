file(GLOB BASE_SRC "*.cpp" "*.hpp" "*.h")
file(GLOB BASE_HEADERS "*.hpp" "*.h")

add_library(base STATIC ${BASE_SRC})
target_link_libraries(base ${Vulkan_LIBRARY})