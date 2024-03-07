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
layout(location = 8) in vec4 shadowCoord;

layout(push_constant) uniform PushConstants {
    uint objectID;
} pc;

// shadow map resolution:
// set the step size to be larger to improve efficiency
const float unit = 1.0 / 2048.0;
const float lightRadius = 5.0f;

vec2 spiralPattern(int k, float radius, int samples) {
    const float goldenAngle = 2.39996323; // Use golden angle for even distribution
    float angle = goldenAngle * float(k);
    float distance = mix(0.0, radius, sqrt(float(k) / float(samples - 1))); // Distribute samples within a radius
    return vec2(cos(angle), sin(angle)) * distance;
}

// return -1 if fully lit
float getAvgBlockerDepth(vec3 coord, float lightDis) {
    // the further the point is away from the light,
    // the larger the area it casts on the shadowMap
    float searchRadius = lightRadius * lightDis / 5000;
    int count = 0;
    float sum = 0;
    // Spiral search pattern
    int numSamples = 500;
    for (int k = 0; k < numSamples; ++k) {
        vec2 offset = spiralPattern(k, searchRadius, numSamples); // Generate offset for sample k
        float sampledDepth = texture(shadowSampler, coord.xy + offset).r;
        if (sampledDepth < coord.z) { 
            count++;
            sum += sampledDepth;
        }
    }
    if (count == 0) {
        return -1.0;
    } else {
        return sum / float(count);
    }
}


layout(location = 0) out vec4 outColor;

float getShadow(vec4 shadowCoord, float lightDis) {
    vec3 projCoord = shadowCoord.xyz / shadowCoord.w;
    projCoord.xy = projCoord.xy * 0.5 + 0.5;
    float blockerDepth = getAvgBlockerDepth(projCoord, lightDis);
    if (blockerDepth < 0) {
        return 1.0;
    } else {
        int count = 0;
        int sum = 0;
        float kernelSize = max((projCoord.z - blockerDepth) * lightRadius / 100.0 / blockerDepth, 0.0);
        for (float i = -kernelSize; i <= kernelSize; i += unit) {
            for (float j = -kernelSize; j <= kernelSize; j += unit) {
                count++;
                if (projCoord.z <= texture(shadowSampler, projCoord.xy + vec2(i, j)).r) {
                    sum++;
               }
            }
        }
        return float(sum) / count;
    }
}


void main() {
    vec3 fragToLight = lightWorldPos - worldPos;
    float distanceSquare = dot(fragToLight, fragToLight);
    vec3 lightDir = normalize(fragToLight);
    vec3 halfDir = normalize(lightDir + normalize(viewPos - worldPos));
    outColor = vec4(texture(texSampler[pc.objectID], texCoord).xyz *
        clamp((
            max(dot(normalize(normal), lightDir), 0.0) * diffuse +
            pow(max(dot(halfDir, normal), 0.0), shininess) * specular * 2.0
            ) / distanceSquare * 200.0 + 0.1, 0.0, 1.0), 1.0)
        * getShadow(shadowCoord, length(fragToLight));
}