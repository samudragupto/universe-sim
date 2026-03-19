#version 460 core

in vec2 vTexCoord;
out vec4 FragColor;

uniform sampler2D uHDRBuffer;

vec3 ACESFilm(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 hdr = texture(uHDRBuffer, vTexCoord).rgb;
    vec3 mapped = ACESFilm(hdr);
    mapped = pow(mapped, vec3(1.0 / 2.2));
    FragColor = vec4(mapped, 1.0);
}