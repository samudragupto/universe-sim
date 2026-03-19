#version 460 core

in vec2 vTexCoord;
out vec4 FragColor;

uniform sampler3D uDensityField;
uniform sampler2D uSceneDepth;
uniform mat4 uInvView;
uniform mat4 uInvProjection;
uniform vec3 uCameraPos;
uniform vec3 uFieldMin;
uniform vec3 uFieldMax;
uniform float uDensityScale;
uniform float uStepSize;
uniform int uMaxSteps;

vec3 getRayDir(vec2 uv) {
    vec4 clip = vec4(uv * 2.0 - 1.0, 1.0, 1.0);
    vec4 eye = uInvProjection * clip;
    eye = vec4(eye.xy, -1.0, 0.0);
    vec3 world = (uInvView * eye).xyz;
    return normalize(world);
}

vec2 intersectBox(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax) {
    vec3 invRd = 1.0 / rd;
    vec3 t0 = (bmin - ro) * invRd;
    vec3 t1 = (bmax - ro) * invRd;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    float tNear = max(max(tmin.x, tmin.y), tmin.z);
    float tFar = min(min(tmax.x, tmax.y), tmax.z);
    return vec2(max(tNear, 0.0), tFar);
}

void main() {
    vec3 ro = uCameraPos;
    vec3 rd = getRayDir(vTexCoord);

    vec2 tHit = intersectBox(ro, rd, uFieldMin, uFieldMax);
    if (tHit.x > tHit.y) {
        FragColor = vec4(0.0);
        return;
    }

    vec3 fieldSize = uFieldMax - uFieldMin;
    float stepSize = uStepSize;
    int maxSteps = uMaxSteps;

    vec4 accum = vec4(0.0);
    float t = tHit.x;

    for (int i = 0; i < maxSteps && t < tHit.y; i++) {
        vec3 pos = ro + rd * t;
        vec3 uvw = (pos - uFieldMin) / fieldSize;

        if (all(greaterThanEqual(uvw, vec3(0.0))) && all(lessThanEqual(uvw, vec3(1.0)))) {
            float density = texture(uDensityField, uvw).r * uDensityScale;

            if (density > 0.001) {
                vec3 emissionColor = vec3(1.0, 0.4, 0.15) * density * 2.0;
                float absorption = density * stepSize * 0.5;
                float transmittance = exp(-absorption);

                accum.rgb += emissionColor * (1.0 - accum.a) * absorption;
                accum.a += (1.0 - accum.a) * (1.0 - transmittance);
            }
        }

        t += stepSize;
        if (accum.a > 0.95) break;
    }

    FragColor = accum;
}