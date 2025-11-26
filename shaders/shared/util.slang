import derivatives;

float length_squared(float3 v) {
    return dot(v, v);
}

// Adapted from http://www.thetenthplanet.de/archives/1180
public float3x3 compute_cotangent_frame(
    float3 normal,
    WithDerivatives<float3> position,
    WithDerivatives<float2> uv
) {
    float3 delta_pos_y_perp = cross(position.dy, normal);
    float3 delta_pos_x_perp = cross(normal, position.dx);

    float3 t = delta_pos_y_perp * uv.dx.x + delta_pos_x_perp * uv.dy.x;
    float3 b = delta_pos_y_perp * uv.dx.y + delta_pos_x_perp * uv.dy.y;

    float invmax = 1.0 / sqrt(max(length_squared(t), length_squared(b)));
    return float3x3(t * invmax, b * invmax, normal);
}

public

func div_ceil<T : IInteger>(T value, uint divisor) -> T {
    return (value + T(divisor - 1)) / T(divisor);
}

public

float2 texel_size(uint2 extent) {
    return 1.0 / float2(extent);
}

public

float2 thread_to_uv(uint2 thread, uint2 extent) {
    return (float2(thread) + 0.5) * texel_size(extent);
}

public

float2 thread_to_clip(uint2 thread, uint2 extent) {
    return thread_to_uv(thread, extent) * 2.0 - 1.0;
}

// https://www.reddit.com/r/GraphicsProgramming/comments/f9zwin/linearising_reverse_depth_buffer/fix7ifb/
public

float linearize_reverse_infinite_depth_buffer(
    float sample_value,
    float near_depth
) {
    return near_depth / sample_value;
}

float3
quadratic_colour_thresholding(float3 colour, float threshold, float knee) {
    float3 curve = float3(threshold - knee, knee * 2.0, 0.25 / knee);

    float brightness = max(colour.x, max(colour.y, colour.z));

    float rq = clamp(brightness - curve.x, 0.0, curve.y);
    rq = curve.z * rq * rq;

    return colour * max(rq, brightness - threshold) / max(brightness, 1.0e-4);
}

struct Ray {
    float3 origin;
    float3 direction;

}

Ray create_camera_ray(float4x4 view_inv, float4x4 proj_inv, float2 uv)
{
    Ray ray;

    let d = uv * 2.0 - 1.0;
    ray.origin = mul(view_inv, float4(0.0, 0.0, 0.0, 1.0)).xyz;
    let temp = mul(proj_inv, float4(d.x, d.y, 1.0, 1.0));
    ray.direction = mul(view_inv, float4(normalize(temp.xyz), 0.0)).xyz;

    return ray;
}

public float3 world_position_from_depth(
    float2 clip,
    float depth,
    float4x4 view_proj_inverse
) {
    // Transform back into world position.
    let world_pos = mul(view_proj_inverse, float4(clip, depth, 1.0));

    // Undo projection.
    return world_pos.xyz / world_pos.w;
}

public float3x3 build_tbn(float3 n) {
    float3 t;
    if (abs(n.z) < 0.999) {
        t = normalize(cross(float3(0, 0, 1), n));
    } else {
        t = normalize(cross(float3(0, 1, 0), n));
    }

    let b = cross(n, t);

    return float3x3(t, b, n);
}
