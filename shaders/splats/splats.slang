import shared.colour;
import shared.util;
import shared.noise;
import fallback_bindings;
import splats.structs;

static const float SH_C0 = 0.28209479177387814;
static const float SH_C1 = 0.4886025119029199;
static const uint DEPTH_MAX = (1u << 28) - 1;


// Dilogarithm Li2(x) = -integral_0^x ln(1-s)/s ds.
// Polynomial fit to Li2(x)-(1-x)ln(1-x), then (1-x)ln(1-x) added back.
float dilog(float x) {
    float y = -6.09201442e-01;
    y = fma(y, x, 1.79126616e+00);
    y = fma(y, x, -2.14953223e+00);
    y = fma(y, x, 1.26304372e+00);
    y = fma(y, x, -4.59069895e-01);
    y = fma(y, x, -1.87417414e-01);
    y = fma(y, x, 1.99603130e+00);
    y = fma(y, x, 5.99669467e-05);
    float s = 1.0 - x;
    if (s > 0.0)
        y += s * log(max(s, 1e-37));
    return y;
}

// Inverse dilogarithm: minimax degree-10 polynomial in t = x / Li2(1).
float inv_dilog(float x) {
    static const float LI2_MAX = 1.6449340668482264; // Li2(1) = pi^2 / 6
    float t = min(x / LI2_MAX, 1.0);
    float y = -1.27463503e-01;
    y = fma(y, t, 5.88993459e+01);
    y = fma(y, t, -1.16025780e+02);
    y = fma(y, t, 1.26945827e+02);
    y = fma(y, t, -8.43108826e+01);
    y = fma(y, t, 3.48799862e+01);
    y = fma(y, t, -8.89606235e+00);
    y = fma(y, t, 1.38640936e+00);
    y = fma(y, t, -7.80640876e-01);
    y = fma(y, t, 1.64841888e+00);
    y = fma(y, t, -2.82836687e-05);
    return y;
}

// Opacity-corrected Box-Muller: returns a 2D standard normal whose radius follows
// the opacity-corrected Gaussian density. Multiply by the Cholesky factor of cov2D.
float2 correctedBoxMuller(float u1, float u2, float alpha) {
    float a = inv_dilog((1.0 - u1) * dilog(alpha)) / max(alpha, 1e-37);
    a = clamp(a, 1e-37, 1.0);
    float r = sqrt(max(-2.0 * log(a), 0.0));
    float theta = 6.2831853 * u2;
    return float2(r * cos(theta), r * sin(theta));
}

// Poisson sampler: Giles (2016) Algorithm 955 QN3 asymptotic approximation.
// Accurate for moderate-to-large means; the guard avoids div-by-zero for tiny means.
uint poisson(float u1, float u2, float lambda) {
    if (lambda < 1e-4)
        return 0;
    float w = sqrt(-2.0 * log(max(u1, 1e-37))) * cos(6.2831853 * u2);
    float w2 = w * w, w3 = w2 * w, w4 = w2 * w2;
    float s = sqrt(lambda);
    float kf = lambda + s * w + (w2 - 1.0) / 6.0;
    kf += (1.0 / s)       * (-(1.0 / 36.0) * w  - (1.0 / 72.0) * w3);
    kf += (1.0 / lambda)  * (-(8.0 / 405.0) + (7.0 / 810.0) * w2 + (1.0 / 270.0) * w4);
    return uint(max(round(kf), 0.0));
}


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
        float3(0.0, focal*invz, -focal*t.y*invz*invz));
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

    // view-space center (same scene-scale as proj_mean) + 2D screen covariance
    let t = mul(pc.view, float4(splat.center * -100, 1.0)).xyz;
    // scale x100 matches the x100 baked into the projection's center; remove both together
    let cov2d = computeCov2D(pc.view, splat.rot, splat.scale * 100.0, t, pc.tan_y, float2(pc.extent));

    // check determinant in case of splats that have a dimension of size 0.
    // Not strictly required for scenes with good splats.

    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;

    if (det <= 0.0) {
        return;
    }

    var rng = SobolRng(thread, pc.frame_index);

    // Cholesky factor of cov2d: maps a standard normal into a sample from this Gaussian
    let c0 = sqrt(cov2d.x);
    let c1 = cov2d.y / c0;
    let c2 = sqrt(max(cov2d.z - c1 * c1, 0.0));

    let center_px = (proj_mean.xy + 1) * 0.5 * pc.extent;

    // opacity-corrected expected point count: lambda = 2*pi*sqrt(det)*Li2(splat.opacity)
    let lambda = 6.2831853 * sqrt(det) * dilog(splat.opacity);
    let pu = rng.rand2();
    let count = poisson(pu.x, pu.y, lambda);

    // per-splat constants: SH base color + reversed-Z depth, packed for nearest-wins InterlockedMax
    let col = (splat.dc * SH_C0) + 0.5;
    let col_u = uint64_t3(col * 255);
    let depth_bits = uint64_t(proj_mean.z * float(DEPTH_MAX));
    let packed = col_u.r | (col_u.g << 8) | (col_u.b << 16) | depth_bits << 36;

    // throw `count` opacity-corrected points; nearest wins per pixel
    for (uint k = 0; k < count; k++) {
        let su = rng.rand2();
        let n = correctedBoxMuller(su.x, su.y, splat.opacity);
        let offset = float2(c0 * n.x, c1 * n.x + c2 * n.y);

        let sample_px = center_px + offset;
        if (any(sample_px < 0.0) || any(sample_px >= pc.extent)) {
            continue;
        }
        let coord = uint2(floor(sample_px));

        InterlockedMax(pc.bitmasks[coord.x + coord.y * pc.extent.x], packed);
    }
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
