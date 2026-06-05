import shared.colour;
import shared.util;
import shared.noise;
import fallback_bindings;
import splats.structs;

static const float SH_C0 = 0.28209479177387814;
static const float SH_C1 = 0.4886025119029199;

// Standard deviation cutoff beyond which a Gaussian is considered negligible (Eq. 2).
static const float NEGLIGIBLE_S2 = (2.0 * sqrt(2.0)) * (2.0 * sqrt(2.0)); // s = 2√2, s² = 8

// Algorithm 1, line 1 — GetGauss1D: project the 3D Gaussian onto the ray.
// Returns the 1D Gaussian mean hit distance `t` (g1.μ) and the squared
// Mahalanobis distance `d2` of the mean. `g.rot` stores the *conjugate*
// (inverse) quaternion, so rotate_quat maps object → local axis-aligned frame,
// where Σ⁻¹ = diag(inv_scale_2). For the ray o + t·d with Δ = o - center:
//   a = dᵀΣ⁻¹d,  b = dᵀΣ⁻¹Δ
//   t = -b/a,     d2 = (ac − b²)/a   [computed as Σ cross products, see below]
void get_gauss_1d(
    Splat g,
    float3 origin,
    float3 dir,
    out float t,
    out float d2,
) {
    float3 delta_l = rotate_quat(g.rot, origin - g.center);
    float3 dir_l   = rotate_quat(g.rot, dir);

    float a = dot(dir_l * dir_l, g.inv_scale_2);
    float b = dot(dir_l * delta_l, g.inv_scale_2);

    // cr = cross(dir_l, delta_l): cr.x = dy·Δz - dz·Δy, cyclic
    float3 cr  = cross(dir_l, delta_l);
    float num  = dot(cr * cr, float3(g.inv_scale_2.y * g.inv_scale_2.z, g.inv_scale_2.z * g.inv_scale_2.x, g.inv_scale_2.x * g.inv_scale_2.y));
    d2 = num / a;
    t  = -b / a;
}

[shader("compute")]
[NumThreads(8, 8, 1)]
void
    write(uint2 thread : SV_DispatchThreadID, uniform PushConstants constants) {
    if (any(thread >= constants.extent)) {
        return;
    }

    let as = RaytracingAccelerationStructure(constants.tlas);

    let d = thread_to_clip(thread, constants.extent);
    let origin = mul(constants.view_inv, float4(0.0, 0.0, 0.0, 1.0)).xyz;
    let temp = mul(constants.proj_inv, float4(d.x, d.y, 1.0, 1.0));
    let direction =
        mul(constants.view_inv, float4(normalize(temp.xyz), 0.0)).xyz;

    RayDesc desc;
    desc.Direction = direction;
    desc.Origin = origin;
    desc.TMax = 10000.0;
    desc.TMin = 0.1;

    var query = RayQuery<RAY_FLAG_NONE>();
    query.TraceRayInline(
        as,
        RAY_FLAG_NONE,
        0xff,
        desc
    );

    var rng = SobolRng(thread, constants.frame_index);

    // Tracked alongside the query so we can test the Gaussian mean against the
    // current closest accepted hit (traversal order is not depth-sorted).
    float tmax = desc.TMax;

    while (query.Proceed()) {
        if (query.CandidateType() != CANDIDATE_PROCEDURAL_PRIMITIVE) {
            continue;
        }

        let g = constants.splats[query.CandidatePrimitiveIndex()];

        // Transform the world-space ray into this primitive's object (BLAS)
        // space, where the splat's center/scale/rot live. The line parameter
        // t is invariant under this affine map (dir_o kept un-normalized), so
        // the committed t is the correct world-space hit distance.
        let w2o = query.CandidateWorldToObject3x4();
        let origin_o = mul(w2o, float4(origin, 1.0));
        let dir_o    = mul(w2o, float4(direction, 0.0));

        // Line 1-2: 1D Gaussian along the ray and its mean depth.
        float t, d2;
        get_gauss_1d(g, origin_o, dir_o, t, d2);

        // Line 3-4: cull if the mean is outside the valid ray range.
        if (t <= desc.TMin || t >= tmax) {
            continue;
        }

        // Line 5-6: cull if negligible (mean outside the sσ ellipsoid, Eq. 2).
        if (d2 > NEGLIGIBLE_S2) {
            continue;
        }

        let alpha = g.opacity_factor * exp(-0.5 * d2);

        if (rng.rand() < alpha) {
            query.CommitProceduralPrimitiveHit(t);
            tmax = t;
        }
    }

    // Shade the closest accepted intersection (the "ray-hit" program).
    float3 value = float3(0);
    if (query.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
        let g = constants.splats[query.CommittedPrimitiveIndex()];

        // Evaluate SH in object space — the frame the coefficients were
        // trained in (so an instance reflection/scale is handled correctly).
        let w2o = query.CommittedWorldToObject3x4();
        let view_o = normalize(mul(w2o, float4(direction, 0.0)));

        float3 sh = g.dc * SH_C0
            + SH_C1 * (-g.sh1_0 * view_o.y + g.sh1_1 * view_o.z - g.sh1_2 * view_o.x);
        value = max(sh + 0.5, 0.0);
    }

    WTexture2D<float4> output = storage_images[constants.image];
    output.Store(thread, float4(srgb_lin2encoded(value), 1));
}
