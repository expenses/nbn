import derivatives;

float length_squared(float3 v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
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

public uint div_ceil(uint value, uint divisor) {
    return (value + divisor - 1) / divisor;
}

public float2 texel_size(uint2 extent) {
    return 1.0 / float2(extent);
}

public float2 thread_to_uv(uint2 thread, uint2 extent) {
    return (float2(thread) + 0.5) * texel_size(extent);
}

// https://www.reddit.com/r/GraphicsProgramming/comments/f9zwin/linearising_reverse_depth_buffer/fix7ifb/
public float linearize_reverse_infinite_depth_buffer(float sample_value, float near_depth) {
    return near_depth / sample_value;
}

float3 quadratic_colour_thresholding(float3 colour, float threshold, float knee) {
    float3 curve = float3(threshold - knee, knee * 2.0, 0.25 / knee);

    float brightness = max(colour.x, max(colour.y, colour.z));

    float rq = clamp(brightness - curve.x, 0.0, curve.y);
    rq = curve.z * rq * rq;

    return colour * max(rq, brightness - threshold) / max(brightness, 1.0e-4);
}
