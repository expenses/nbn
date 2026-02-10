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
    let basis = abs(n.z) < 0.5 ? float3(0, 0, 1) : float3(0, 1, 0);
    let t = normalize(cross(basis, n));
    let b = cross(n, t);
    return transpose(float3x3(t, b, n));
}

public float inv_lerp(float value, float from, float to){
  return (value - from) / (to - from);
}

// See Ray Tracing Gems II, Chapter 4, HACKING THE SHADOW TERMINATOR
public float3 shadow_terminator_hack(
    float3 P,
    float3 A,
    float3 B,
    float3 C,
    float3 nA,
    float3 nB,
    float3 nC,
    float3 barycentrics
) {
    let u = barycentrics[0];
    let v = barycentrics[1];
    let w = barycentrics[2];

    // get distance vectors from triangle vertices
    var tmpu = P - A;
    var tmpv = P - B;
    var tmpw = P - C;
    // project these onto the tangent planes
    // defined by the shading normals
    float dotu = min(0.0, dot(tmpu , nA));
    float dotv = min(0.0, dot(tmpv , nB));
    float dotw = min(0.0, dot(tmpw , nC));
    tmpu -= dotu*nA;
    tmpv -= dotv*nB;
    tmpw -= dotw*nC;
    // finally P' is the barycentric mean of these three
    let Pp = P + u*tmpu + v*tmpv + w*tmpw;
    return Pp;
}
