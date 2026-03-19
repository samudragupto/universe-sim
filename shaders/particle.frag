#version 460 core
in vec4 vColor; in float vDist; out vec4 FragColor;
void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0; float r2 = dot(coord, coord);
    if (r2 > 1.0) discard;
    float alpha = exp(-r2 * 5.0) + exp(-r2 * 1.2) * 0.25;
    if (alpha * vColor.a < 0.001) discard;
    FragColor = vec4(vColor.rgb * alpha, alpha * vColor.a);
}