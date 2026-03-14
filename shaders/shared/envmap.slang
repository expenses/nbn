import constants;

float2 envmap_direction_to_uv(float3 direction) {
    return float2(
        atan2(direction.z, direction.x) / (2.0 * PI) + 0.5,
        acos(direction.y) / PI
    );
}

float3 envmap_uv_to_direction(float2 uv) {
    let phi   = uv.x * 2.0 * PI;
    let theta = uv.y * PI;
    return float3(
        sin(theta) * cos(phi),
        cos(theta),
        sin(theta) * sin(phi),
    );
}
