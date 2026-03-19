#version 460 core
in vec2 vTexCoord; out vec4 FragColor; uniform sampler2D uHDR; uniform float uThreshold;
void main() {
    vec3 c = texture(uHDR, vTexCoord).rgb;
    FragColor = vec4(dot(c, vec3(0.2126, 0.7152, 0.0722)) > uThreshold ? c : vec3(0), 1.0);
}