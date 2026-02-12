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
    uint triangle_index;
};

[shader("vertex")] Interp vertex(uint vertex_index: SV_VertexID, uniform PushConstants push_constants) {
    let index = push_constants.model.read_index(vertex_index);
    let uv2 = push_constants.model.uv2s[index];

    Interp interp;
    interp.uv2_pos = float4(uv2 * 2.0 - 1.0, 0.0, 1.0);
    interp.triangle_index = vertex_index / 3;
    return interp;
}

[shader("pixel")] uint fragment(Interp in) {
    return in.triangle_index;
}

// 2D cross product
float cross2d(float2 a, float2 b) {
    return a.x * b.y - a.y * b.x;
}

[shader("compute")]
[NumThreads(8,8,1)]
void lightmap(uint2 thread : SV_DispatchThreadID, uniform PushConstants push_constants) {
    if (any(thread >= push_constants.extent)) {
        return;
    }

    Texture2D<uint> visbuffer = storage_images[push_constants.visbuffer];
    let triangle_index = visbuffer[thread];
    if (triangle_index == MAX_UINT_VALUE) {
        return;
    }

    let model = push_constants.model;

    let indices = model.read_indices(triangle_index);
    let coord_uv = thread_to_uv(thread, push_constants.extent);
    let a = model.uv2s[indices.x];
    let b = model.uv2s[indices.y];
    let c = model.uv2s[indices.z];
    let denom = cross2d(b - a, c - a);
    var barycentrics = float2(
        cross2d(coord_uv - a, c - a),
        cross2d(b - a, coord_uv - a)
    ) / denom;
    barycentrics = clamp(barycentrics, float2(0), float2(1));

    let pos = interpolate_from_buffer(model.positions, indices, barycentrics);
    let normal = normalize(interpolate_from_buffer(model.normals, indices, barycentrics));

    let pushed_pos = shadow_terminator_hack(
        pos,
        model.positions[indices.x],
        model.positions[indices.y],
        model.positions[indices.z],
        model.normals[indices.x],
        model.normals[indices.y],
        model.normals[indices.z],
        float3(1.0-barycentrics.x-barycentrics.y, barycentrics)
    );

    let tlas = RaytracingAccelerationStructure(push_constants.tlas);

    var accum = 0;

    let tbn = build_tbn(normal);


    for (uint i = 0; i < push_constants.samples_per_iter; i++) {
        let rand2 = frac(push_constants.blue_noise_data.sample2(thread, 0) + R2(push_constants.sample_index + i));

        let hemisphere_dir = sample_cosine_weighted_hemisphere(rand2);
        let rotated_dir = mul(tbn, hemisphere_dir);

        RayDesc desc;
        desc.Direction = rotated_dir;
        desc.Origin = pushed_pos;
        desc.TMax = 200.0;
        desc.TMin = 1e-4;

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

    RWTexture2D<float4> output = storage_images[push_constants.output];
    output[thread].rgb += float3(float(accum) / float(push_constants.total_samples));
    output[thread].a = 1.0;
}
