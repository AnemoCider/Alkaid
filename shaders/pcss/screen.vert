#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat3 normalRot;
    mat4 lightSpace;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec2 fragTexCoord;
layout (location = 3) out vec2 fragShadowCoord;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragNormal =  ubo.normalRot * inNormal;
    fragTexCoord = inTexCoord;
    fragShadowCoord = (ubo.lightSpace * vec4(inPosition, 1.0)).st;	
}