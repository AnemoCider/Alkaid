#version 450

layout(binding = 1) uniform sampler2D texSampler[2];
layout(binding = 2) uniform sampler2D shadowSampler;

layout(location = 0) in vec3 normal;
layout(location = 1) in vec3 worldPos;
layout(location = 2) in vec2 texCoord;
layout(location = 3) in vec3 diffuse;
layout(location = 4) in vec3 specular;
layout(location = 5) in float shininess;
layout(location = 6) in vec3 lightWorldPos;
layout(location = 7) in vec3 viewPos;

layout(push_constant) uniform PushConstants {
    uint objectID;
} pc;


layout(location = 0) out vec4 outColor;


void main() {
    vec3 fragToLight = lightWorldPos - worldPos;
    float distanceSquare = dot(fragToLight, fragToLight);
    vec3 lightDir = normalize(fragToLight);
    vec3 halfDir = normalize(lightDir + normalize(viewPos - worldPos));
    outColor = vec4(texture(texSampler[pc.objectID], texCoord).xyz *
        clamp((
            max(dot(normalize(normal), lightDir), 0.0) * diffuse + 
            pow(max(dot(halfDir, normal), 0.0), shininess) * specular * 2.0
        ) / distanceSquare * 3.0 + 0.1, 0.0, 1.0), 1.0);
    // outColor = texture(texSampler[pc.objectID], texCoord);
}