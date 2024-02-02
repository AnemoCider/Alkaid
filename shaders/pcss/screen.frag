#version 450


layout(binding = 1) uniform sampler2D texSampler;
layout(binding = 2) uniform sampler2D shadownSampler;

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(texSampler, fragTexCoord);
}