import constants;
import random;

float3 sample_cosine_weighted_hemisphere(float2 u) {
    let phi = 2.0 * PI * u.x;
    let radius = sqrt(u.y);

    let x = radius * cos(phi);
    let y = radius * sin(phi);
    let z = sqrt(1 - u.y);

    return float3(x, y, z);
}

float3 sample_cosine_weighted_hemisphere<R : Rng>(R* rng) {
    return sample_cosine_weighted_hemisphere(rng.rand2());
}

float3 sample_uniform_hemisphere(float2 u) {
    let phi = 2.0 * PI * u.x;
    let z = 1.0 - u.y;
    let radius = sqrt(1.0 - z * z);
    let x = radius * cos(phi);
    let y = radius * sin(phi);
    return float3(x, y, z);
}

float3 sample_uniform_hemisphere<R : Rng>(R* rng) {
    return sample_uniform_hemisphere(rng.rand2());
}

float3 sample_sphere(float2 rand) {
    let z = 1.0 - 2.0 * rand.x;
    let phi = 2.0 * PI * rand.y;
    let r = sqrt(1.0 - z * z);

    return float3(r * cos(phi), r * sin(phi), z);
}

float3 sample_sphere<R : Rng>(R* rng) {
    return sample_sphere(rng.rand2());
}

// The following is taken from ray tracing gems II

// rect.x and rect.y are between 0 and 1.
float2 MapRectToCircle(float2 rect) {
    let radius = sqrt(rect.x);
    let angle = rect.y * 2.0 * PI;
    return float2(radius * cos(angle), radius * sin(angle));
}

// rect.x and rect.y are between 0 and 1. direction is normalized direction light. radius could be ~0.1.
float3 SphericalDirectionalLightRayDirection(
    float2 rect,
    float3 direction,
    float radius
) {
    let point = MapRectToCircle(rect) * radius;

    let cross_dir = direction.y > 0.5 ? float3(0, 0, 1) : float3(0, 1, 0);
    
    let tangent = normalize(cross(direction, cross_dir));
    let bitangent = normalize(cross(tangent, direction));
    return normalize(direction + point.x * tangent + point.y * bitangent);
}
