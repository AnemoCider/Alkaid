#version 450

layout(binding = 1) uniform samplerCube cubeMap;

layout(location = 0) in vec3 texCoord;

layout(location = 0) out vec4 outColor;

void main() {
    // outColor = texture(cubeMap, texCoord);
    //if (isinf(texCoord.x) || isinf(texCoord.y) || isinf(texCoord.z) ||
        /*isnan(texCoord.x) || isnan(texCoord.y) || isnan(texCoord.z)) {
        res = vec3(1.0);
    }*/
    outColor = vec4(1.0, 1.0, 1.0, 1.0);
}