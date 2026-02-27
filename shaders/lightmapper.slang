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

    let loc = thread.y * push_constants.extent.x + thread.x;
    let location_bitmask = push_constants.location_bitmasks[loc];

    if (location_bitmask == 0) {
        return;
    }

    let tlas = RaytracingAccelerationStructure(push_constants.tlas);
    var accum = float3(0);

    for (uint i = 0; i < push_constants.samples_per_iter; i++) {
        let rand = push_constants.rand2(thread, 1, push_constants.sample_index + i).x;

        var tri_query = RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH>();
        hit_uv_tlas(thread, random_bit(location_bitmask, rand), push_constants, tri_query);
        tri_query.Proceed();

        if (tri_query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
            // For debugging. Shouldn't occur under normal circumstances.
            push_constants.output[loc] = float4(1,0,0,1);
            return;
        }

        let rand2 = push_constants.rand2(thread, 0, push_constants.sample_index + i);
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
    }

    push_constants.temp[loc].rgb += accum / float(push_constants.total_samples);
    push_constants.temp[loc].a = 1.0;
}

void hit_uv_tlas(
    uint2 thread,
    uint i,
    PushConstants push_constants,
    inout RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> tri_query
) {
    let uv_tlas = RaytracingAccelerationStructure(push_constants.uv_tlas);

    let uv_ray_offset = frac(0.5 + R2(i));

    RayDesc tri_desc;
    tri_desc.Origin = float3(
        (float2(thread)+uv_ray_offset)/push_constants.extent,
        1
    );
    tri_desc.Direction = float3(0,0,-1);
    tri_desc.TMax = 2.0;
    tri_desc.TMin = 0;

    tri_query.TraceRayInline(
        uv_tlas,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        0xff,
        tri_desc
    );
    tri_query.Proceed();
}

bool check_for_backfaces(float2 barycentrics, uint triangle_index, PushConstants push_constants) {
    let tlas = RaytracingAccelerationStructure(push_constants.tlas);

    let indices = push_constants.model.read_indices(triangle_index);

    let a = push_constants.model.positions[indices.x];
    let b = push_constants.model.positions[indices.y];
    let c = push_constants.model.positions[indices.z];
    let geom_normal = normalize(cross(b - a, c - a));

    let pos = offset_ray(interpolate_from_buffer(push_constants.model.positions, indices, barycentrics), geom_normal);

    float3[4] dirs = float3[4](float3(1,0,0), float3(0,1,0), float3(-1,0,0), float3(0,-1,0));

    for (uint i = 0; i < 4; i++) {
        RayDesc desc;
        desc.Origin = pos;
        desc.Direction = mul(build_tbn(geom_normal), dirs[i]);
        desc.TMax = 1;
        desc.TMin = 0;

        var query = RayQuery<RAY_FLAG_NONE>();
        query.TraceRayInline(
            tlas,
            RAY_FLAG_NONE,
            0xff,
            desc
        );
        query.Proceed();

        if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT && !query.CommittedRayFrontFace()) {
            return true;
        }
    }

    return false;
}

[shader("compute")]
[NumThreads(8,8,1)]
void check_locations(uint2 thread : SV_DispatchThreadID, uniform PushConstants push_constants) {
    if (any(thread >= push_constants.extent)) {
        return;
    }

    let uv_tlas = RaytracingAccelerationStructure(push_constants.uv_tlas);
    let loc = thread.y * push_constants.extent.x + thread.x;

    uint64_t bitmask = 0;

    for (uint i = 0; i < 64; i++) {
        var tri_query = RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH>();
        hit_uv_tlas(thread, i, push_constants, tri_query);
        tri_query.Proceed();
        var hit = tri_query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
        if (hit) {
            hit = !check_for_backfaces(tri_query.CommittedRayBarycentrics(), tri_query.CommittedPrimitiveIndex(), push_constants);
        }
        bitmask |= uint64_t(hit) << i;
    }

    push_constants.location_bitmasks[loc] = bitmask;
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

float4 dilate(float4* values, uint2 coord, uint2 extent) {
    var value = values[coord.y * extent.x + coord.x];

    if (value.a > 0.5) {
        return value;
    }

    int2[8] corners = int2[8](
        int2(1,0), int2(-1,0), int2(0,1), int2(0,-1),
        int2(1,1), int2(-1,1), int2(1,-1), int2(-1,-1)
    );

    value = float4(0);
    var count = 0;

    for (uint i = 0; i < 8; i++) {
        let sample_pos = int2(coord)+corners[i];
        if (all(sample_pos >= 0) && all(sample_pos < extent)) {
            let sample = values[sample_pos.y * extent.x + sample_pos.x];
            if (sample.a > 0.5) {
                value += sample;
                count++;
            }
        }
    }

    if (count > 0) {
        return value / count;
    }

    return float4(0);
}

[shader("compute")]
[NumThreads(8,8,1)]
void dilation(uint2 thread : SV_DispatchThreadID, uniform PushConstants push_constants) {
    if (any(thread >= push_constants.extent)) {
        return;
    }

    let loc = thread.y * push_constants.extent.x + thread.x;
    push_constants.output[loc] = dilate(push_constants.temp, thread, push_constants.extent);
}

[shader("compute")]
[NumThreads(8,8,1)]
void render(uint2 thread: SV_DispatchThreadID, uniform ViewerConstants constants) {
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
    desc.TMax = 200.0;
    desc.TMin = 0.1;

    var query = RayQuery<RAY_FLAG_NONE >();
    query.TraceRayInline(
        as,
        RAY_FLAG_NONE,
        0xff,
        desc
    );

    query.Proceed();

    var value = float3(0.0);

    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        let barycentrics = query.CommittedTriangleBarycentrics();
        value = float3(1.0 - barycentrics.x - barycentrics.y, barycentrics);
    }

    WTexture2D<float4> output = storage_images[constants.image];
    output.Store(thread, float4(srgb_lin2encoded(value), 1));
}
