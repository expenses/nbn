import lightmapper_structs;
import fallback_bindings;

import shared.noise;
import shared.sampling;
import shared.util;
import shared.colour;
import shared.constants;
import shared.barycentrics;
import shared.noise;

[shader("compute")]
[NumThreads(8,8,1)]
void lightmap(uint2 thread : SV_DispatchThreadID, uniform PushConstants push_constants) {
    if (any(thread >= push_constants.extent)) {
        return;
    }

    let tlas = RaytracingAccelerationStructure(push_constants.tlas);
    let uv_tlas = RaytracingAccelerationStructure(push_constants.uv_tlas);
    let loc = thread.y * push_constants.extent.x + thread.x;

    var accum = float3(0);

    for (uint i = 0; i < push_constants.samples_per_iter; i++) {
        let uv_ray_offset = frac(push_constants.blue_noise_data.sample2(thread, 1) + R2(push_constants.sample_index + i));

        RayDesc tri_desc;
        tri_desc.Origin = float3(
            (float2(thread)+uv_ray_offset)/push_constants.extent,
            1
        );
        tri_desc.Direction = float3(0,0,-1);
        tri_desc.TMax = 2.0;
        tri_desc.TMin = 0;

        var tri_query = RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH>();
        tri_query.TraceRayInline(
            uv_tlas,
            RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
            0xff,
            tri_desc
        );
        tri_query.Proceed();

        if (tri_query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
            continue;
        }

        let rand2 = frac(push_constants.blue_noise_data.sample2(thread, 0) + R2(push_constants.sample_index + i));
        let desc = trace_from(tri_query.CommittedRayBarycentrics(), tri_query.CommittedPrimitiveIndex(), rand2, push_constants);

        var query = RayQuery<RAY_FLAG_NONE>();
        query.TraceRayInline(
            tlas,
            RAY_FLAG_NONE,
            0xff,
            desc
        );
        query.Proceed();

        let missed = query.CommittedStatus() != COMMITTED_TRIANGLE_HIT;

        accum += float(missed);
        push_constants.valid[loc] += missed || query.CommittedRayFrontFace();
    }

    push_constants.temp[loc].rgb += accum / float(push_constants.total_samples);
    push_constants.temp[loc].a = 1.0;
}

RayDesc trace_from(float2 barycentrics, uint triangle_index, float2 rand2, PushConstants push_constants) {
    let indices = push_constants.model.read_indices(triangle_index);
    let a = push_constants.model.positions[indices.x];
    let b = push_constants.model.positions[indices.y];
    let c = push_constants.model.positions[indices.z];

    let geom_normal = normalize(cross(b - a, c - a));

    let pos = offset_ray(shadow_terminator_hack(
        a, b, c,
        push_constants.model.normals[indices.x],
        push_constants.model.normals[indices.y],
        push_constants.model.normals[indices.z],
        float3(1.0-barycentrics.x-barycentrics.y, barycentrics)
    ), geom_normal);

    let normal = normalize(interpolate_from_buffer(push_constants.model.normals, indices, barycentrics));
    let tbn = build_tbn(normal);

    let hemisphere_dir = sample_cosine_weighted_hemisphere(rand2);
    let rotated_dir = mul(tbn, hemisphere_dir);

    RayDesc desc;
    desc.Direction = rotated_dir;
    desc.Origin = pos;
    desc.TMax = 200.0;
    desc.TMin = 0;

    return desc;
}

[shader("compute")]
[NumThreads(8,8,1)]
void dilation(uint2 thread : SV_DispatchThreadID, uniform PushConstants push_constants) {
    if (any(thread >= push_constants.extent)) {
        return;
    }

    let loc = thread.y * push_constants.extent.x + thread.x;
    push_constants.output[loc] = push_constants.temp[loc];
}

[shader("compute")]
[NumThreads(8,8,1)]
void downsampling(uint2 thread : SV_DispatchThreadID, uniform PushConstants push_constants) {
    if (any(thread >= push_constants.extent)) {
        return;
    }

    let loc = thread.y * push_constants.extent.x + thread.x;

    let fraction = float(push_constants.valid[loc]) / float(push_constants.total_samples);

    if (fraction > 0.5) {
        push_constants.output[loc].rgb = push_constants.temp[loc].rgb / fraction;
        push_constants.output[loc].a = 1.0;
    }
}
