#version 460 core
layout(location=0) in vec3 aPos; layout(location=1) in vec4 aCol; layout(location=2) in float aSize;
uniform mat4 uView; uniform mat4 uProjection; uniform float uScreenHeight;
out vec4 vColor; out float vDist;
void main() {
    vec4 viewPos = uView * vec4(aPos, 1.0); gl_Position = uProjection * viewPos;
    vDist = length(viewPos.xyz);
    gl_PointSize = clamp((aSize * uScreenHeight) / (1.0 + 0.05 * vDist), 1.0, 64.0);
    vColor = vec4(aCol.rgb / (1.0 + vDist * 0.002), aCol.a / (1.0 + vDist * 0.002));
}