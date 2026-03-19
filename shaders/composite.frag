#version 460 core
in vec2 vTexCoord; out vec4 FragColor;
uniform sampler2D uScene, uBloom, uVol; uniform float uInt, uExp; uniform bool uHasVol;
vec3 aces(vec3 x) { return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14), 0.0, 1.0); }
void main() {
    vec3 c = texture(uScene, vTexCoord).rgb + texture(uBloom, vTexCoord).rgb * uInt;
    if(uHasVol) { vec4 v = texture(uVol, vTexCoord); c = c * (1.0 - v.a) + v.rgb; }
    FragColor = vec4(pow(aces(c * uExp), vec3(1.0/2.2)), 1.0);
}