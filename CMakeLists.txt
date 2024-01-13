cmake_minimum_required(VERSION 3.4 FATAL_ERROR)

project(Alkaid)

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_VERSION VERSION_LESS 3.7.0)
	message(STATUS "Using module to find Vulkan")
	find_package(Vulkan)
endif()

include_directories(external)
include_directories(external/vulkan/include)
include_directories(external/glfw/include)
include_directories(source/base)

add_subdirectory(source/base)
add_subdirectory(external/glfw)

add_executable(Renderer source/examples/basicTriangle.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_WIN32_KHR")

target_link_libraries(Renderer base glfw ${Vulkan_LIBRARY})