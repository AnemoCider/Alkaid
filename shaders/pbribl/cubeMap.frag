#version 450

layout(location = 0) in vec3 normal;
layout(location = 1) in vec3 worldPos;
layout(location = 2) in vec3 viewPos;

layout(push_constant) uniform PushConsts {
	float roughness;
	float metallic;
	float specular;
	float r;
	float g;
	float b;
	uint index;
} material;

layout(binding = 1) uniform sampler2D samplerBRDFLUT;
layout(binding = 2) uniform samplerCube prefilteredMap;

layout(location = 0) out vec4 outColor;

#define PI 3.1415926535897932384626433832795
#define ALBEDO vec3(material.r, material.g, material.b)

// Normal Distribution function --------------------------------------
float D_GGX(float dotNH, float roughness) {
	float alpha = roughness * roughness;
	float alpha2 = alpha * alpha;
	float denom = dotNH * dotNH * (alpha2 - 1.0) + 1.0;
	return (alpha2) / (PI * denom * denom);
}

// Geometric Shadowing function --------------------------------------
float G_SchlicksmithGGX(float dotNL, float dotNV, float roughness) {
	float r = (roughness + 1.0);
	float k = (r * r) / 8.0;
	float GL = dotNL / (dotNL * (1.0 - k) + k);
	float GV = dotNV / (dotNV * (1.0 - k) + k);
	return GL * GV;
}

// Fresnel function ----------------------------------------------------

vec3 F_SchlickR(float cosTheta, vec3 F0, float roughness) {
	return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}

vec3 prefilteredReflection(vec3 R, float roughness) {
	// totalMipCount = log2(dim_of_map) = 9
	const float MAX_REFLECTION_LOD = 9.0;
	float lod = roughness * MAX_REFLECTION_LOD;
	float lodf = floor(lod);
	float lodc = ceil(lod);
	// trilinear interpolation
	vec3 a = textureLod(prefilteredMap, R, lodf).rgb;
	vec3 b = textureLod(prefilteredMap, R, lodc).rgb;
	return mix(a, b, lod - lodf);
}

void main() {
	vec3 N = normalize(normal);
	vec3 V = normalize(viewPos - worldPos);
	vec3 R = reflect(-V, N);
	R.x = -R.x;

	float metallic = material.metallic;
	float roughness = material.roughness;

	vec3 F0 = vec3(0.04);
	F0 = mix(F0, ALBEDO, metallic);

	vec2 brdf = texture(samplerBRDFLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
	vec3 reflection = prefilteredReflection(R, roughness).rgb;

	vec3 F = F_SchlickR(max(dot(N, V), 0.0), F0, roughness);

	vec3 color = reflection * (F * brdf.x + brdf.y);

	outColor = vec4(color, 1.0);
}