cmake_minimum_required(VERSION 3.4 FATAL_ERROR)

project(Alkaid)

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_VERSION VERSION_LESS 3.7.0)
	message(STATUS "Using module to find Vulkan")
	find_package(Vulkan)
endif()

include_directories(external)
include_directories(external/ktx/include)
include_directories(external/ktx/other_include)
include_directories(external/vulkan/include)
include_directories(external/tinyglTF)
include_directories(external/glfw/include)
include_directories(external/glm)
include_directories(external/tinygltf)
include_directories(source/base)


add_subdirectory(source/base)
add_subdirectory(external/glfw)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_WIN32_KHR")

# Compile Shaders

set(GLSL_COMPILER ${CMAKE_CURRENT_SOURCE_DIR}/shaders/glslc.exe)
set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

function(buildExample EXAMPLE_NAME)
    set(EXAMPLE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/source/examples/${EXAMPLE_NAME})
    message(STATUS "Generating project file for example in ${EXAMPLE_FOLDER}")
    add_executable(${EXAMPLE_NAME} source/examples/basicTriangle.cpp)
    target_link_libraries(${EXAMPLE_NAME} base glfw ${Vulkan_LIBRARY})

    file(GLOB SHADER_SOURCES CONFIGURE_DEPENDS "${SHADER_DIR}/${EXAMPLE_NAME}/*.vert" "${SHADER_DIR}/${EXAMPLE_NAME}/*.frag" )
    # Iterate over each shader file
    foreach(SHADER ${SHADER_SOURCES})
        # Get the shader file name without extension
        get_filename_component(SHADER_NAME ${SHADER} NAME)
        # Define the compiled shader file name with .spv extension
        set(SHADER_SPIRV "${SHADER_DIR}/${EXAMPLE_NAME}/${SHADER_NAME}.spv")
        # Define the command to compile the shader into .spv format
        add_custom_command(
            OUTPUT ${SHADER_SPIRV}
            COMMAND ${GLSL_COMPILER} ${SHADER} -o ${SHADER_SPIRV} 
            DEPENDS ${SHADER}
            COMMENT "Compiling ${SHADER}"
        )
        # Store all the compiled shaders in a list
        list(APPEND SPIRV_BINARY_FILES ${SHADER_SPIRV})
    endforeach()
    add_custom_target(
        Shaders 
        DEPENDS ${SPIRV_BINARY_FILES} 
        SOURCES ${SHADER_SOURCES}
    )
    add_dependencies(${EXAMPLE_NAME} Shaders)

endfunction(buildExample)

buildExample(basicTriangle)