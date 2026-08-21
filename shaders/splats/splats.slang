import shared.colour;
import shared.util;
import shared.noise;
import fallback_bindings;
import splats.structs;
import shared.prefix_sum;
import shared.constants;
import splats.helpers;

static const float GAUSSIAN_TRUNCATION_STD = 3.0;
static const uint K = 8;
static const uint NUM_SPLATS_PER_CHUNK = 80 * 1000 * 1000;


// i-th column of the rotation matrix: where the quaternion sends basis vector i
float3 rotatedBasis(float4 q, int i) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    if (i == 0) return float3(1-2*(y*y+z*z), 2*(x*y+z*w),   2*(x*z-y*w));
    if (i == 1) return float3(2*(x*y-z*w),   1-2*(x*x+z*z), 2*(y*z+x*w));
    return       float3(2*(x*z+y*w),         2*(y*z-x*w),   1-2*(x*x+y*y));
}

// Jacobian of perspective projection: view-space vector -> screen-pixel delta (2x3)
float2x3 screenJacobian(float3 t, float focal) {
    float invz = 1.0 / t.z;
    return float2x3(
        float3(focal*invz, 0.0, -focal*t.x*invz*invz),
        float3(0.0, -focal*invz, focal*t.y*invz*invz));
}

// 2D screen covariance as (a,b,c) for [[a,b],[b,c]].
// Camera intrinsics are derived from tan_y + the screen extent (symmetric perspective,
// so focal is isotropic: focal = extent.y / (2*tan_y)).
float3 computeCov2D(float4x4 view, float4 q, float3 scale, float3 t,
                    float tan_y, float2 extent) {
    float aspect = extent.x / extent.y;
    float tan_x  = tan_y * aspect;
    float focal  = extent.y / (2.0 * tan_y);

    // clamp the center's view angle to ~FOV (EWA stability near the edges)
    t.x = clamp(t.x/t.z, -1.3*tan_x, 1.3*tan_x)*t.z;
    t.y = clamp(t.y/t.z, -1.3*tan_y, 1.3*tan_y)*t.z;

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
[NumThreads(1, 1, 1)]
void reset(uniform PushConstants pc) {
    pc.dispatch.num_points = 0;
}

[shader("compute")]
[NumThreads(64, 1, 1)]
void splat(uint thread : SV_DispatchThreadID, uniform PushConstants pc) {
    if (thread >= pc.num_splats) {
        return;
    }

    let chunk = pc.splats[thread / NUM_SPLATS_PER_CHUNK];
    let splat = chunk[thread % NUM_SPLATS_PER_CHUNK];

    var proj_mean = mul(pc.camera, float4(splat.center, 1.0));
    proj_mean.xyz /= proj_mean.w;

    if (proj_mean.z < 0) {
        return;
    }

    // Potential speed up by culling splat centers if they're _way_ offscreen.
    if (abs(proj_mean.x) > 2 || abs(proj_mean.y) > 2) {
        return;
    }

    // view-space center (same scene-scale as proj_mean) + 2D screen covariance
    let t = mul(pc.view, float4(splat.center, 1.0)).xyz;
    // scale x100 matches the x100 baked into the projection's center; remove both together
    var cov2d = computeCov2D(pc.view, splat.rot, splat.scale, t, pc.tan_y, float2(pc.extent));
    cov2d += float3(0.3, 0, 0.3);

    // check determinant in case of splats that have a dimension of size 0.
    // Not strictly required for scenes with good splats.

    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;

    if (det <= 0.0) {
        return;
    }

    // Cholesky factor of cov2d: maps a standard normal into a sample from this Gaussian
    let c0 = sqrt(cov2d.x);
    let c1 = cov2d.y / c0;
    let c2 = sqrt(max(cov2d.z - c1 * c1, 0.0));

    let center_px = (proj_mean.xy + 1) * 0.5 * pc.extent;

    let half_ext = GAUSSIAN_TRUNCATION_STD * sqrt(float2(cov2d.x, cov2d.z));
    if (any(center_px + half_ext < 0.0) || any(center_px - half_ext >= pc.extent)) {
        return;
    }

    let importance = TWO_PI * sqrt(det) * dilog(splat.opacity);

    uint2 seed = makeSeed(thread, pc.frame_index);
    var count = poisson(seed, importance/K);

    if (count == 0) {
        return;
    }

    count = min(count, 20000);

    uint offset = 0;
    InterlockedAdd(pc.dispatch.num_points, count, offset);

    for (uint i = offset; i < min(offset+count, pc.max_points); i++) {
        pc.point_to_splat[i] = thread;
    }
}

[shader("compute")]
[NumThreads(1,1,1)]
void setup_dispatch(uniform PushConstants pc) {
    if (pc.dispatch.num_points > pc.max_points) {
        printf("too many points: %u\n", pc.dispatch.num_points);
    }
    pc.dispatch.num_points = min(pc.dispatch.num_points, pc.max_points);
    let total = pc.dispatch.num_points;
    pc.dispatch.dispatch_size = uint3(div_ceil(total, 64), 1, 1);
}

[shader("compute")]
[NumThreads(64, 1, 1)]
void point(uint thread : SV_DispatchThreadID, uniform PushConstants pc) {
    if (thread >= pc.dispatch.num_points) {
        return;
    }

   	uint2 seed = makeSeed(thread, pc.frame_index);

    let splat_index = pc.point_to_splat[thread];
    let chunk = pc.splats[splat_index / NUM_SPLATS_PER_CHUNK];
    let splat = chunk[splat_index % NUM_SPLATS_PER_CHUNK];

    var proj_mean = mul(pc.camera, float4(splat.center, 1.0));
    proj_mean.xyz /= proj_mean.w;

    // view-space center (same scene-scale as proj_mean) + 2D screen covariance
    let t = mul(pc.view, float4(splat.center, 1.0)).xyz;
    // scale x100 matches the x100 baked into the projection's center; remove both together
    var cov2d = computeCov2D(pc.view, splat.rot, splat.scale, t, pc.tan_y, float2(pc.extent));
    cov2d += float3(0.3, 0, 0.3);

    // Cholesky factor of cov2d: maps a standard normal into a sample from this Gaussian
    let c0 = sqrt(cov2d.x);
    let c1 = cov2d.y / c0;
    let c2 = sqrt(max(cov2d.z - c1 * c1, 0.0));

    let depth_bits = uint64_t(reinterpret<uint>(clamp(proj_mean.z, 0.0, 1.0)));

    let packed = splat.color_opacity | (depth_bits << 32);

    let center_px = (proj_mean.xy + 1) * 0.5 * pc.extent;

    for (uint i = 0; i < K; i++) {
        let u = pcg2d(seed);
        let xy = correctedBoxMuller(u.x, u.y, splat.opacity);
        var pixel = float2(center_px.x + c0 * xy.x, center_px.y + c1 * xy.x + c2 * xy.y);
        pixel = floor(pixel);

        if (any(pixel < 0.0) || any(pixel >= pc.extent)) {
            continue;
        }
        let coord = uint2(pixel);

        // reject points whose contribution is below the 8-bit floor (Mahalanobis test)
        float2 delta = center_px - pixel;
        let z0 = delta.x / c0;
        let z1 = (delta.y - c1 * z0) / c2;
        let gaussian_weight = exp(-0.5 * (z0*z0 + z1*z1));
        let alpha = splat.opacity * gaussian_weight;
        if (alpha < 1.0 / 255.0) {
            continue;
        }

        InterlockedMax(*pc.get_mask(coord), packed);
    }
}

[shader("compute")]
[NumThreads(8, 8, 1)]
void output(uint2 thread : SV_DispatchThreadID, uniform PushConstants pc) {
    if (any(thread >= pc.extent)) {
        return;
    }

    let srgb_color = pc.srgb_color_and_depth(thread).rgb;
    *pc.get_mask(thread) = 0;

    pc.image.Store(thread, srgb_color);
}
