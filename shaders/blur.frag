#version 460 core
in vec2 vTexCoord; out vec4 FragColor; uniform sampler2D uImg; uniform bool uHoriz;
const float w[5] = float[](0.227, 0.194, 0.121, 0.054, 0.016);
void main() {
    vec2 off = 1.0 / textureSize(uImg, 0); vec3 res = texture(uImg, vTexCoord).rgb * w[0];
    for(int i=1; i<5; ++i) {
        vec2 d = uHoriz ? vec2(off.x*i, 0) : vec2(0, off.y*i);
        res += texture(uImg, vTexCoord + d).rgb * w[i]; res += texture(uImg, vTexCoord - d).rgb * w[i];
    }
    FragColor = vec4(res, 1.0);
}