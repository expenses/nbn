import lightmapper_structs;
import fallback_bindings;

import shared.noise;
import shared.sampling;
import shared.util;
import shared.colour;
import shared.constants;
import shared.barycentrics;
import shared.noise;

struct Interp {
    float4 uv2_pos: SV_Position;
    float3 position;
    float3 normal;
    uint triangle_index;
};

[shader("vertex")] Interp vertex(uint vertex_index: SV_VertexID, uniform PushConstants push_constants) {
    let index = push_constants.model.read_index(vertex_index);
    let uv2 = push_constants.model.uv2s[index];

    Interp interp;
    interp.uv2_pos = float4(uv2 * 2.0 - 1.0, 0.0, 1.0);
    interp.triangle_index = vertex_index / 3;
    interp.position = push_constants.model.positions[index];
    interp.normal = push_constants.model.normals[index];
    return interp;
}

Optional<float3> hit_backface(float3 position, float3 unnormalized_dir, PushConstants push_constants) {
    let as = RaytracingAccelerationStructure(push_constants.tlas);

    RayDesc desc;
    desc.Origin = position;
    desc.Direction = unnormalized_dir;
    // Setting this to 1.0 results in the point being pushed too far sometimes.
    // 0.9 probably isn't a perfect value but it's better so whatever.
    desc.TMax = 0.9;
    desc.TMin = 0;

    var query = RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH>();
    query.TraceRayInline(
        as,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        0xff,
        desc
    );
    query.Proceed();

    if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT || query.CommittedRayFrontFace()) {
        return none;
    }

    let triangle_index = query.CommittedPrimitiveIndex();
    let barycentrics = query.CommittedTriangleBarycentrics();

    let indices = push_constants.model.read_indices(triangle_index);

    let a = push_constants.model.positions[indices.x];
    let b = push_constants.model.positions[indices.y];
    let c = push_constants.model.positions[indices.z];

    let normal = normalize(cross(b - a, c - a));

    let position = interpolate_ray_barycentrics(a, b, c, barycentrics);

    return offset_ray(position, normal);
}

struct Output {
    float4 position: COLOR0;
    float3 normal: COLOR1;
}

[shader("pixel")] Output fragment(Interp in, float3 barycentrics: SV_Barycentrics, uniform PushConstants push_constants) {
    let ddx = ddx(in.position);
    let ddy = ddy(in.position);

    Output output;
    output.normal = normalize(in.normal);

    //barycentrics.yz = clamp(barycentrics.yz, float2(0), float2(1));
    //barycentrics.x = 1.0 - barycentrics.y - barycentrics.z;

    let indices = push_constants.model.read_indices(in.triangle_index);

    let a = push_constants.model.positions[indices.x];
    let b = push_constants.model.positions[indices.y];
    let c = push_constants.model.positions[indices.z];

    let geom_normal = normalize(cross(b - a, c - a));

    output.position = float4(offset_ray(shadow_terminator_hack(
        a, b, c,
        push_constants.model.normals[indices.x],
        push_constants.model.normals[indices.y],
        push_constants.model.normals[indices.z],
        barycentrics
    ), geom_normal), 1.0);

    float3[4] dirs = float3[4](ddx, ddy, -ddx, -ddy);

    for (uint i = 0; i < 4; i++) {
        if (let position = hit_backface(output.position.xyz, dirs[i], push_constants)) {
            output.position.xyz = position;
            break;
        }
    }

    return output;
}

[shader("compute")]
[NumThreads(8,8,1)]
void lightmap(uint2 thread : SV_DispatchThreadID, uniform PushConstants push_constants) {
    if (any(thread >= push_constants.extent)) {
        return;
    }

    Texture2D<float4> positions_buffer = storage_images[push_constants.positions];
    Texture2D<float3> normal_buffer = storage_images[push_constants.normals];

    let pos_value = positions_buffer[thread];

    if (pos_value.a < 0.5) {
        return;
    }

    let pos = pos_value.xyz;
    let normal = normal_buffer[thread];
    let tbn = build_tbn(normal);

    let tlas = RaytracingAccelerationStructure(push_constants.tlas);

    var accum = 0;

    for (uint i = 0; i < push_constants.samples_per_iter; i++) {
        let rand2 = frac(push_constants.blue_noise_data.sample2(thread, 0) + R2(push_constants.sample_index + i));

        let hemisphere_dir = sample_cosine_weighted_hemisphere(rand2);
        let rotated_dir = mul(tbn, hemisphere_dir);

        RayDesc desc;
        desc.Direction = rotated_dir;
        desc.Origin = pos;
        desc.TMax = 200.0;
        desc.TMin = 0;

        var query = RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH>();
        query.TraceRayInline(
            tlas,
            RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
            0xff,
            desc
        );
        query.Proceed();

        accum += uint(query.CommittedStatus() != COMMITTED_TRIANGLE_HIT);
    }

    RWTexture2D<float4> temp = storage_images[push_constants.temp];
    temp[thread].rgb += float3(float(accum) / float(push_constants.total_samples));
    temp[thread].a = 1.0;
}

[shader("compute")]
[NumThreads(8,8,1)]
void dilation(uint2 thread : SV_DispatchThreadID, uniform PushConstants push_constants) {
    if (any(thread >= push_constants.extent)) {
        return;
    }

    Texture2D<float4> temp = storage_images[push_constants.temp];
    WTexture2D<float4> output = storage_images[push_constants.output];
    output.Store(thread, dilate(temp, thread, push_constants.extent));
}
