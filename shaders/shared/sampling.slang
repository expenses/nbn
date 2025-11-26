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

float3 sample_cosine_weighted_hemisphere<R: Rng>(R* rng) {
    return sample_cosine_weighted_hemisphere(float2(rng.rand(), rng.rand()));
}
