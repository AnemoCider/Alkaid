#version 450

layout(location = 0) in vec3 normal;
layout(location = 1) in vec3 worldPos;
layout(location = 2) in vec2 texCoord;
layout(location = 3) in vec3 lightPos;
layout(location = 4) in vec3 viewPos;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 fragToLight = lightPos - worldPos;
    float distanceSquare = dot(fragToLight, fragToLight);
    vec3 lightDir = normalize(fragToLight);
    vec3 halfDir = normalize(lightDir + normalize(viewPos - worldPos));
    outColor = vec4(vec3(0.8, 0.8, 0.8) *
        clamp((
            max(dot(normalize(normal), lightDir), 0.0) * vec3(0.8, 0.8, 0.8) + 
            pow(max(dot(halfDir, normal), 0.0), 15) * vec3(1.0) * 2.0
        ) / distanceSquare * 15.0 + 0.1, 0.0, 1.0), 1.0);
    // outColor = texture(texSampler[pc.objectID], texCoord);
}