#version 460 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aCol;
layout(location=2) in float aSize;

uniform mat4 uView;
uniform mat4 uProjection;
uniform float uScreenHeight;

out vec4 vColor;

void main() {
    vec4 viewPos = uView * vec4(aPos, 1.0);
    gl_Position = uProjection * viewPos;

    float dist = max(length(viewPos.xyz), 0.001);
    gl_PointSize = clamp(aSize * 200.0 / dist, 2.0, 12.0);

    vColor = aCol;
}