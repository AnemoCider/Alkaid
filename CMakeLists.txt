cmake_minimum_required(VERSION 3.29)

project(Alkaid VERSION 2.0
                DESCRIPTION "Alkaid Experimental"
                LANGUAGES CXX)

# ==================================================================================================
# General Pre-build Settings
# ==================================================================================================

### Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
        STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# ==================================================================================================
#  Dependencies
# ==================================================================================================

find_package(Vulkan REQUIRED)

# ==================================================================================================
#  Paths
# ==================================================================================================

set(EXTERNAL ${CMAKE_CURRENT_SOURCE_DIR}/extern)

# ==================================================================================================
# Sub-projects
# ==================================================================================================

# ==================================================================================================
# Main project
# ==================================================================================================

add_library(Renderer STATIC alkaid/src/main.cpp)

target_link_libraries(Renderer PRIVATE Vulkan::Vulkan)

# ==================================================================================================
# Target Properties
# ==================================================================================================

target_compile_features(Renderer PUBLIC cxx_std_20)
set_target_properties(Renderer PROPERTIES CXX_EXTENSIONS OFF)