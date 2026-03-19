#version 460 core
in vec2 vTexCoord; out vec4 FragColor; uniform sampler2D uScene; uniform float uVig, uChroma;
void main() {
    vec2 c = vec2(0.5), d = vTexCoord - c; float dist = length(d) * 1.414;
    float vig = clamp(1.0 - dist * dist * uVig, 0.0, 1.0);
    float r = texture(uScene, vTexCoord + d * uChroma).r;
    float g = texture(uScene, vTexCoord).g;
    float b = texture(uScene, vTexCoord - d * uChroma).b;
    FragColor = vec4(vec3(r,g,b) * vig, 1.0);
}