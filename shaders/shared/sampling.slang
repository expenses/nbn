
// Take 13 samples in a grid around the center pixel:
// . . . . . . .
// . A . B . C .
// . . D . E . .
// . F . G . H .
// . . I . J . .
// . K . L . M .
// . . . . . . .
// These samples are interpreted as 4 overlapping boxes
// plus a center box.
float3 sample_13_tap_box_filter(
    Sampler2D<float3> texture,
    float2 uv,
    float2 texel_size
) {
    float3 a = texture.SampleLevel(uv + texel_size * float2(-1, -1), 0);
    float3 b = texture.SampleLevel(uv + texel_size * float2(0, -1), 0);
    float3 c = texture.SampleLevel(uv + texel_size * float2(1, -1), 0);

    float3 d = texture.SampleLevel(uv + texel_size * float2(-0.5, -0.5), 0);
    float3 e = texture.SampleLevel(uv + texel_size * float2(0.5, -0.5), 0);

    float3 f = texture.SampleLevel(uv + texel_size * float2(-1, 0), 0);
    float3 g = texture.SampleLevel(uv, 0);
    float3 h = texture.SampleLevel(uv + texel_size * float2(1, 0), 0);

    float3 i = texture.SampleLevel(uv + texel_size * float2(-0.5, 0.5), 0);
    float3 j = texture.SampleLevel(uv + texel_size * float2(0.5, 0.5), 0);

    float3 k = texture.SampleLevel(uv + texel_size * float2(-1, 1), 0);
    float3 l = texture.SampleLevel(uv + texel_size * float2(0, 1), 0);
    float3 m = texture.SampleLevel(uv + texel_size * float2(1, 1), 0);

    float3 center_pixels = d + e + i + j;

    float3 top_left = a + b + f + g;
    float3 top_right = b + c + g + h;
    float3 bottom_left = f + g + k + l;
    float3 bottom_right = g + h + l + m;

    return center_pixels * 0.25 * 0.5
        + (top_left + top_right + bottom_left + bottom_right) * 0.25 * 0.125;
}

// Sample in a 3x3 grid but with weights to produce a tent filter:
//
//        a*1 b*2 c*1
// 1/16 * d*2 e*4 f*2
//        g*1 h*2 i*1
float3 sample_3x3_tent_filter(
    Sampler2D<float3> texture,
    float2 uv,
    float2 texel_size
) {
    float3 a = texture.SampleLevel(uv + texel_size * float2(-1, -1), 0);
    float3 b = texture.SampleLevel(uv + texel_size * float2(0, -1), 0);
    float3 c = texture.SampleLevel(uv + texel_size * float2(1, -1), 0);

    float3 d = texture.SampleLevel(uv + texel_size * float2(-1, 0), 0);
    float3 e = texture.SampleLevel(uv, 0);
    float3 f = texture.SampleLevel(uv + texel_size * float2(1, 0), 0);

    float3 g = texture.SampleLevel(uv + texel_size * float2(-1, 1), 0);
    float3 h = texture.SampleLevel(uv + texel_size * float2(0, 1), 0);
    float3 i = texture.SampleLevel(uv + texel_size * float2(1, 1), 0);

    return ((a + c + g + i) + (b + d + f + h) * 2.0 + e * 4.0) / 16.0;
}

// The following is taken from ray tracing gems II

// rect.x and rect.y are between 0 and 1.
float2 MapRectToCircle(float2 rect) {
    let radius = sqrt(rect.x);
    let angle = rect.y * 2.0 * 3.14159265359;
    return float2(radius * cos(angle), radius * sin(angle));
}

// rect.x and rect.y are between 0 and 1. direction is normalized direction light. radius could be ~0.1.
public

float3 SphericalDirectionalLightRayDirection(
    float2 rect,
    float3 direction,
    float radius
) {
    let point = MapRectToCircle(rect) * radius;
    let tangent = normalize(cross(direction, float3(0.0, 1.0, 0.0)));
    let bitangent = normalize(cross(tangent, direction));
    return normalize(direction + point.x * tangent + point.y * bitangent);
}

// rect.x and rect.y are between 0 and 1. surfacePos and lightPos are in world space. worldRadius is in world units and
// could be ~5.
float3 SphericalPositionalLightRayDirection(
    float2 rect,
    float3 surfacePos,
    float3 lightPos,
    float worldRadius
) {
    let direction = normalize(lightPos - surfacePos);
    let radius = worldRadius / length(lightPos - surfacePos);

    let point = MapRectToCircle(rect) * radius;
    let tangent = normalize(cross(direction, float3(0.0, 1.0, 0.0)));
    let bitangent = normalize(cross(tangent, direction));
    return normalize(direction + point.x * tangent + point.y * bitangent);
}

// rect.x and rect.y are between 0 and 1. surfacePos and lightPos are in world space. worldRadius is in world units and
// could be ~5. angleAtten will be between 0 and 1.
float3 SphericalSpotLightRayDirection(
    float2 rect,
    float3 surfacePos,
    float3 lightPos,
    float worldRadius,
    float3 shineDir,
    float cosThetaInner,
    float cosThetaOuter,
    out float angleAtten
) {
    let direction = normalize(lightPos - surfacePos);
    let radius = worldRadius / length(lightPos - surfacePos);

    angleAtten = dot(direction, -shineDir);
    angleAtten = smoothstep(cosThetaOuter, cosThetaInner, angleAtten);

    let point = MapRectToCircle(rect) * radius;
    let tangent = normalize(cross(direction, float3(0.0, 1.0, 0.0)));
    let bitangent = normalize(cross(tangent, direction));
    return normalize(direction + point.x * tangent + point.y * bitangent);
}

// x and y are in pixels.
public

float interleaved_gradient_noise(int x, int y) {
    return fract(
        52.9829189f * fract(0.06711056f * float(x) + 0.00583715f * float(y))
    );
}

float AnimateBlueNoise(in float blueNoise, in int frameIndex) {
    return fract(blueNoise + float(frameIndex % 32) * 0.61803399);
}
