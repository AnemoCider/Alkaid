cmake_minimum_required(VERSION 3.7 FATAL_ERROR)

project(Alkaid)

set(CMAKE_CXX_STANDARD 20)

if (NOT CMAKE_VERSION VERSION_LESS 3.7.0)
	message(STATUS "Using module to find Vulkan")
	find_package(Vulkan)
endif()

include_directories(external)
include_directories(external/vulkan/include)
include_directories(external/ktx/include)
include_directories(external/ktx/other_include)
include_directories(external/tinyglTF)
include_directories(external/glfw/include)
include_directories(external/glm)
include_directories(external/tinygltf)

include_directories(src/function)

add_executable(myTest src/main.cpp)

add_subdirectory(external/glfw)
add_subdirectory(src/function)

if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_WIN32_MVK")
elseif(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_MACOS_KHR")
endif()

target_link_libraries(myTest glfw ${Vulkan_LIBRARY})
