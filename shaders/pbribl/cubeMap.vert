#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 normalRot;
    vec4 viewPos;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec3 fragWorldPos;
layout(location = 2) out vec3 fragViewPos;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragNormal = (ubo.normalRot * vec4(inNormal, 1.0)).xyz;
    fragWorldPos = (ubo.model * vec4(inPosition, 1.0)).xyz;
    fragViewPos = ubo.viewPos.xyz;
}