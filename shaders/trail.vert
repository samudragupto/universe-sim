#version 460 core
layout(location=0) in vec3 aPos; layout(location=1) in vec4 aCol;
uniform mat4 uView; uniform mat4 uProjection; out vec4 vColor;
void main() { gl_Position = uProjection * uView * vec4(aPos, 1.0); vColor = aCol; }