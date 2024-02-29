#version 450

layout(binding = 1) uniform samplerCube cubeMap;

layout(location = 0) in vec3 texCoord;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(cubeMap, texCoord);
}