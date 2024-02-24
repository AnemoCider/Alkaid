#version 450


layout(binding = 1) uniform sampler2D texSampler;
layout(binding = 2) uniform sampler2D shadowSampler;

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 inDiffuse;
layout(location = 3) in vec3 inSpecular;
layout(location = 4) in float inShininess;

// layout(location = 3) in vec2 fragShadowCoord;

layout(location = 0) out vec4 outColor;


void main() {
    outColor = texture(texSampler, fragTexCoord);
    // outColor = texture(shadowSampler, fragTexCoord);
}