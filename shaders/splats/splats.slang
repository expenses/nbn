import shared.colour;
import shared.util;
import shared.noise;
import fallback_bindings;
import splats.structs;
import shared.prefix_sum;

static const uint DEPTH_MAX = (1u << 30) - 1;
static const float GAUSSIAN_TRUNCATION_STD = 3.0;
static const uint MAX_SPLATS = 20 * 1000 * 1000;
static const uint MAX_POINTS = 100 * 1000 * 1000;


// Dilogarithm Li2(x) = -integral_0^x ln(1-s)/s ds.
// Polynomial fit to Li2(x)-(1-x)ln(1-x), then (1-x)ln(1-x) added back.
float dilog(float x) {
    float y = -6.09201442e-01f;
	y = fma(y, x, 1.79126616e+00f);
	y = fma(y, x, -2.14953223e+00f);
	y = fma(y, x, 1.26304372e+00f);
	y = fma(y, x, -4.59069895e-01f);
	y = fma(y, x, -1.87417414e-01f);
	y = fma(y, x, 1.99603130e+00f);
	y = fma(y, x, 5.99669467e-05f);

	float s = 1.0f - x;
	if (s > 0.0f)
		y += s * log(max(s, 1e-37f));

	return y;
}

// Li2(1) = pi^2 / 6
static const float LI2_MAX = 1.6449340668482264;

// Inverse dilogarithm: minimax degree-10 polynomial in t = x / Li2(1).
float inv_dilog(float x) {
    float t = min(x / LI2_MAX, 1.0f);

	float y = -1.27463503e+01f;
	y = fma(y, t, 5.88993459e+01f);
	y = fma(y, t, -1.16025780e+02f);
	y = fma(y, t, 1.26945827e+02f);
	y = fma(y, t, -8.43108826e+01f);
	y = fma(y, t, 3.48799862e+01f);
	y = fma(y, t, -8.89606235e+00f);
	y = fma(y, t, 1.38640936e+00f);
	y = fma(y, t, -7.80640876e-01f);
	y = fma(y, t, 1.64841888e+00f);
	y = fma(y, t, -2.82836687e-05f);

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
// gaussian-point-splatting/src/core/random/poisson.cuh
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
    pc.dispatch.surviving_splats = 0;
    pc.dispatch.num_points = 0;
}

[shader("compute")]
[NumThreads(64, 1, 1)]
void splat(uint thread : SV_DispatchThreadID, uniform PushConstants pc) {
    if (thread >= pc.num_splats) {
        return;
    }

    let instance_scale = float3(1, -1, 1);
    let instance_translation = float3(0);

    let splat = pc.splats[thread];

    let instance_center = splat.center * instance_scale + instance_translation;

    var proj_mean = mul(pc.camera, float4(instance_center, 1.0));
    proj_mean.xyz /= proj_mean.w;

    if (proj_mean.z < 0) {
        return;
    }

    // view-space center (same scene-scale as proj_mean) + 2D screen covariance
    let t = mul(pc.view, float4(instance_center, 1.0)).xyz;
    // scale x100 matches the x100 baked into the projection's center; remove both together
    let cov2d = computeCov2D(pc.view, splat.rot, splat.scale * instance_scale, t, pc.tan_y, float2(pc.extent));

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

    var rng = SobolRng(thread, pc.frame_index);

    // opacity-corrected expected point count: lambda = 2*pi*sqrt(det)*Li2(splat.opacity)
    let lambda = 6.2831853 * sqrt(det) * dilog(splat.opacity);
    let pu = rng.rand2();
    var count = poisson(pu.x, pu.y, lambda);

    count = min(count, 1000);

    if (count == 0) {
        return;
    }

    uint index = 0;
    InterlockedAdd(pc.dispatch.surviving_splats, 1, index);

    if (index >= MAX_SPLATS) {
        return;
    }

    let depth_bits = uint64_t(proj_mean.z * float(DEPTH_MAX));

    PerSplatData data;
    data.packed = splat.color_opacity | (depth_bits << 32);
    data.c0 = c0;
    data.c1 = c1;
    data.c2 = c2;
    data.center_px = center_px;
    pc.splat_data[index] = data;

    uint offset = 0;
    InterlockedAdd(pc.dispatch.num_points, count, offset);

    for (uint i = offset; i < min(offset+count, MAX_POINTS); i++) {
        pc.point_to_splat[i] = index;
    }
}

[shader("compute")]
[NumThreads(1,1,1)]
void setup_dispatch(uniform PushConstants pc) {
    if (pc.dispatch.surviving_splats > MAX_SPLATS) {
        printf("too many splats: %u\n", pc.dispatch.surviving_splats);
    }
    if (pc.dispatch.num_points > MAX_POINTS) {
        printf("too many points: %u\n", pc.dispatch.num_points);
    }
    pc.dispatch.num_points = min(pc.dispatch.num_points, MAX_POINTS);
    let total = pc.dispatch.num_points;
    pc.dispatch.dispatch_size = uint3(div_ceil(total, 64), 1, 1);
}

[shader("compute")]
[NumThreads(64, 1, 1)]
void point(uint thread : SV_DispatchThreadID, uniform PushConstants pc) {
    if (thread >= pc.dispatch.num_points) {
        return;
    }

    let data = pc.splat_data[pc.point_to_splat[thread]];

    var rng = SobolRng(thread, pc.frame_index);

    let su = rng.rand2();
    let n = correctedBoxMuller(su.x, su.y, data.opacity);
    let offset = float2(data.c0 * n.x, data.c1 * n.x + data.c2 * n.y);

    let sample_px = data.center_px + offset;
    if (any(sample_px < 0.0) || any(sample_px >= pc.extent)) {
        return;
    }
    let coord = uint2(floor(sample_px));

    // reject points whose contribution is below the 8-bit floor (Mahalanobis test)
    let delta = data.center_px - float2(coord);
    let z0 = delta.x / data.c0;
    let z1 = (delta.y - data.c1 * z0) / data.c2;
    let weight = exp(-0.5 * (z0*z0 + z1*z1));
    if (data.opacity * weight < 1.0 / 255.0) {
        return;
    }

    InterlockedMax(pc.bitmasks[coord.x + coord.y * pc.extent.x], data.packed);
}

[shader("compute")]
[NumThreads(8, 8, 1)]
void output(uint2 thread : SV_DispatchThreadID, uniform PushConstants pc) {
    if (any(thread >= pc.extent)) {
        return;
    }

    let mask = pc.bitmasks[thread.x + thread.y * pc.extent.x];
    pc.bitmasks[thread.x + thread.y * pc.extent.x] = 0;

    let value = float3(mask & 255, (mask >> 8) & 255, (mask >> 16) & 255) / 255;

    WTexture2D<float4> output = storage_images[pc.image];
    output.Store(thread, float4(srgb_lin2encoded(value), 1));
}
