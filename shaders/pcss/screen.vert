#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 normalRot;
    vec4 lightPos;
    vec4 viewPos;
    mat4 lightVP;
    vec4 lightFov;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inDiffuse;
layout(location = 4) in vec3 inSpecular;
layout(location = 5) in float inShininess;
layout(location = 6) in int inIllum;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec3 fragWorldPos;
layout(location = 2) out vec2 fragTexCoord;
layout(location = 3) out vec3 fragDiffuse;
layout(location = 4) out vec3 fragSpecular;
layout(location = 5) out float fragShininess;
layout(location = 6) out vec3 fragLightPos;
layout(location = 7) out vec3 fragViewPos;
layout(location = 8) out vec4 fragShadowCoord;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragNormal = (ubo.normalRot * vec4(inNormal, 1.0)).xyz;
    fragWorldPos = (ubo.model * vec4(inPosition, 1.0)).xyz;
    fragTexCoord = inTexCoord;
    fragDiffuse = inDiffuse;
    fragSpecular = inSpecular;
    fragShininess = inShininess;
    if (inIllum < 2) {
        fragSpecular = vec3(0.0, 0.0, 0.0);
    } else {
        fragSpecular = inSpecular;
    }
    fragLightPos = ubo.lightPos.xyz;
    fragViewPos = ubo.viewPos.xyz;
    vec3 normalizedNormal = normalize(inNormal);
    vec3 vecToLight = ubo.lightPos.xyz - inPosition;
    // add a normal bias
    // 128 here is experimental value
    fragShadowCoord = ubo.lightVP * ubo.model * vec4(
        inPosition + normalizedNormal * length(vecToLight) * ubo.lightFov.x / 128.0 * 
        (1 - dot(normalizedNormal, normalize(vecToLight)))
        , 1.0);
    // fragShadowCoord = ubo.lightVP * ubo.model * vec4(inPosition, 1.0);
}