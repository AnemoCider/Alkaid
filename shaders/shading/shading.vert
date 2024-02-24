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
layout(location = 3) in vec3 inDiffuse;
layout(location = 4) in vec3 inSpecular;
layout(location = 5) in float inShininess;
layout(location = 6) in int inIllum;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec3 fragDiffuse;
layout(location = 3) out vec3 fragSpecular;
layout(location = 4) out float fragShininess;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragNormal =  ubo.normalRot * inNormal;
    fragTexCoord = inTexCoord;
    fragDiffuse = inDiffuse;
    fragSpecular = inSpecular;
    fragShininess = inShininess;
    if (inIllum < 2) {
        fragSpecular = vec3(0.0, 0.0, 0.0);
    } else {
        fragSpecular = inSpecular;
    }
}