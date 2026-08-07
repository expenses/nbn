import shared.colour;
import shared.util;
import shared.noise;
import fallback_bindings;
import splats.structs;

static const float SH_C0 = 0.28209479177387814;
static const float SH_C1 = 0.4886025119029199;
static const uint DEPTH_MAX = (1u << 28) - 1;

// i-th column of the rotation matrix: where the quaternion sends basis vector i
float3 rotatedBasis(float4 q, int i) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    if (i == 0) return float3(1-2*(y*y+z*z), 2*(x*y+z*w),   2*(x*z-y*w));
    if (i == 1) return float3(2*(x*y-z*w),   1-2*(x*x+z*z), 2*(y*z+x*w));
    return       float3(2*(x*z+y*w),         2*(y*z-x*w),   1-2*(x*x+y*y));
}

// Jacobian of perspective projection: view-space vector -> screen-pixel delta (2x3)
float2x3 screenJacobian(float3 t, float2 focal) {
    float invz = 1.0 / t.z;
    return float2x3(
        float3(focal.x*invz, 0.0, -focal.x*t.x*invz*invz),
        float3(0.0, focal.y*invz, -focal.y*t.y*invz*invz));
}

// 2D screen covariance as (a,b,c) for [[a,b],[b,c]].
float3 computeCov2D(float4x4 view, float4 q, float3 scale, float3 t,
                    float2 tan_fov, float2 focal) {
    // clamp the center's view angle to ~FOV (EWA stability near the edges)
    float limx = 1.3*tan_fov.x, limy = 1.3*tan_fov.y;
    t.x = clamp(t.x/t.z, -limx, limx)*t.z;
    t.y = clamp(t.y/t.z, -limy, limy)*t.z;

    float3x3 W = float3x3(view[0].xyz, view[1].xyz, view[2].xyz);  // world -> view
    float2x3 P = mul(screenJacobian(t, focal), W);                 // world -> screen

    float3 cov = 0.0;
    [unroll] for (int i = 0; i < 3; i++) {
        float2 s = mul(P, rotatedBasis(q, i)) * scale[i];          // axis in screen space
        cov += float3(s.x*s.x, s.x*s.y, s.y*s.y);                  // += outer product
    }
    return cov;
}

[shader("compute")]
[NumThreads(64, 1, 1)]
void splat(uint thread : SV_DispatchThreadID, uniform PushConstants pc) {
    if (thread >= pc.num_splats) {
        return;
    }

    let splat = pc.splats[thread];

    var proj_mean = mul(pc.camera, float4(splat.center * -100, 1.0));
    proj_mean.xyz /= proj_mean.w;

    if (any(abs(proj_mean.xy) > 1) || proj_mean.z < 0) {
        return;
    }

    let coord = uint2((proj_mean.xy + 1) / 2.0 * pc.extent);

    let col = (splat.dc * SH_C0) + 0.5;
    let col_u = uint64_t3(col * 255);

    let depth_bits = uint64_t(proj_mean.z * float(DEPTH_MAX));

    InterlockedMax(
        pc.bitmasks[coord.x + coord.y * pc.extent.x],
        col_u.r | (col_u.g << 8) | (col_u.b << 16) | depth_bits << 36
    );
}

[shader("compute")]
[NumThreads(8, 8, 1)]
void output(uint2 thread : SV_DispatchThreadID, uniform PushConstants pc) {
    if (any(thread >= pc.extent)) {
        return;
    }

    let mask = pc.bitmasks[thread.x + thread.y * pc.extent.x];

    let value = float3(mask & 255, (mask >> 8) & 255, (mask >> 16) & 255) / 255;

    WTexture2D<float4> output = storage_images[pc.image];
    output.Store(thread, float4(srgb_lin2encoded(value), 1));
}
