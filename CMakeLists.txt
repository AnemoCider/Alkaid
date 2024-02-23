cmake_minimum_required(VERSION 3.7 FATAL_ERROR)

project(Alkaid)

set(CMAKE_CXX_STANDARD 20)

if (NOT CMAKE_VERSION VERSION_LESS 3.7.0)
	message(STATUS "Using module to find Vulkan")
	find_package(Vulkan)
endif()

# Set an empty list for subdirs to add source files

include_directories(external)
include_directories(external/vulkan/include)
include_directories(external/ktx/include)
include_directories(external/ktx/other_include)
include_directories(external/tinyglTF)
include_directories(external/glfw/include)
include_directories(external/glm)
include_directories(external/tinygltf)

include_directories(src/function)

add_subdirectory(external/glfw)
add_subdirectory(src/function)

if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_WIN32_MVK")
elseif(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_MACOS_KHR")
endif()

set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")   
set(GLSL_COMPILER "${SHADER_DIR}/glslc.exe")

function(buildExample EXAMPLE_NAME)
    set(EXAMPLE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/src/examples/${EXAMPLE_NAME}/)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${EXAMPLE_FOLDER})
    message(STATUS "Generating project file for example in ${EXAMPLE_FOLDER}")

    file(GLOB SHADER_SOURCES "${SHADER_DIR}/${EXAMPLE_NAME}/*.vert" "${SHADER_DIR}/${EXAMPLE_NAME}/*.frag" )

    add_executable(${EXAMPLE_NAME} ${EXAMPLE_FOLDER}/${EXAMPLE_NAME}.cpp)

    # Iterate over each shader file
    foreach(SHADER ${SHADER_SOURCES})
        # Get the shader file name without extension
        get_filename_component(SHADER_NAME ${SHADER} NAME)
        # Define the compiled shader file name with .spv extension
        set(SHADER_SPIRV "${SHADER_DIR}/${EXAMPLE_NAME}/${SHADER_NAME}.spv")
        # Define the command to compile the shader into .spv format
        add_custom_command(
            TARGET ${EXAMPLE_NAME}
            PRE_BUILD
            # COMMAND cd ${SHADER_DIR}
            COMMAND ${GLSL_COMPILER} ${SHADER} -o ${SHADER_SPIRV} 
            DEPENDS ${SHADER}
            COMMENT "Compiling ${SHADER}"
            WORKING_DIRECTORY ${SHADER_DIR}
        )

        # Store all the compiled shaders in a list
        list(APPEND SPIRV_BINARY_FILES ${SHADER_SPIRV})
    endforeach()

    target_sources(${EXAMPLE_NAME} PRIVATE ${SHADER_SOURCES} )
    target_link_libraries(${EXAMPLE_NAME} base glfw)
    source_group(Shaders FILES ${SHADER_SOURCES})
    message(STATUS "Runtime executables written to ${EXAMPLE_FOLDER}")
endfunction(buildExample)

buildExample(basicTriangle)
# buildExample(pcss)

