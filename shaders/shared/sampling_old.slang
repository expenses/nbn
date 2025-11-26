import constants;

// The following is taken from ray tracing gems II

// rect.x and rect.y are between 0 and 1.
float2 MapRectToCircle(float2 rect) {
    let radius = sqrt(rect.x);
    let angle = rect.y * 2.0 * PI;
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

float AnimateBlueNoise(in float blueNoise, in int frameIndex) {
    return fract(blueNoise + float(frameIndex % 32) * 0.61803399);
}
