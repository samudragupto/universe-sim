#version 460 core
in vec4 vColor;
out vec4 FragColor;

void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(p, p);
    if (r2 > 1.0) discard;

    // Soft Gaussian falloff for the star core
    float core = exp(-r2 * 6.0);
    
    // Wider, fainter halo
    float halo = exp(-r2 * 2.0) * 0.3;
    
    float alpha = core + halo;
    
    // Preserve the original color, boost the core brightness slightly
    vec3 finalColor = vColor.rgb * (1.0 + core * 0.5);

    FragColor = vec4(finalColor * alpha, alpha * vColor.a);
}