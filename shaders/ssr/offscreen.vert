#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 lightMVP;
} ubo;

layout(location = 0) in vec3 inPosition;

void main() {
    gl_Position = ubo.lightMVP * vec4(inPosition, 1.0);
}