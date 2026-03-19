#version 460 core
in vec2 vTexCoord; out vec4 FragColor;
uniform sampler2D uScene; uniform int uBHCount; uniform vec2 uBHPos[16]; 
uniform float uBHM[16]; uniform float uBHR[16]; uniform vec2 uRes;
void main() {
    vec2 uv = vTexCoord, totOff = vec2(0);
    for(int i=0; i<uBHCount && i<16; i++) {
        vec2 d = uv - uBHPos[i]; float dist = length(d * uRes); float r = uBHR[i];
        if(dist < r*10.0 && dist > r*0.1) {
            totOff += normalize(d) * (uBHM[i] / (dist*dist + 1.0)) * 0.01;
            if(abs(dist - r*1.5) < r*0.3) {
                FragColor = texture(uScene, uv - totOff);
                FragColor.rgb += vec3(1.0, 0.9, 0.6) * exp(-abs(dist-r*1.5)) * uBHM[i] * 0.001; return;
            }
        }
    }
    FragColor = texture(uScene, uv - totOff);
}