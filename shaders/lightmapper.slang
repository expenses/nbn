import lightmapper_structs;
import shared.noise;
import shared.sampling;
import shared.util;
import shared.colour;
import shared.constants;
import fallback_bindings;

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

[shader("compute")]
[NumThreads(8,8,1)]
void lightmap(uint2 thread : SV_DispatchThreadID, uniform PushConstants push_constants) {
    if (any(thread >= push_constants.extent)) {
        return;
    }

    Texture2D<uint> visbuffer = storage_images[push_constants.visbuffer];
    let value = visbuffer[thread];
    if (value == MAX_UINT_VALUE) {
        return;
    }

    RWTexture2D<float4> output = storage_images[push_constants.output];

    output[thread] = float4(colour_from_val(value), 1.0);
}

/*
[shader("pixel")] float4 fragment2(Interp in, uniform PushConstants push_constants) {
    let tlas = RaytracingAccelerationStructure(push_constants.tlas);
    let shading_normal = normalize(in.shading_normal);

    var rand = ThreadBlueNoise(push_constants.blue_noise_data, uint2(in.uv2_pos.xy), 0, 0);
    var accum = 0;

    let tbn = build_tbn(shading_normal);

    for (uint i = 0; i < 128; i++) {
        let hemisphere_dir = sample_cosine_weighted_hemisphere(&rand);
        let rotated_dir = mul(tbn, hemisphere_dir);

        RayDesc desc;
        desc.Direction = rotated_dir;
        desc.Origin = in.pos;
        desc.TMax = 200.0;
        desc.TMin = 1e-5;

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

    return float4(float3(float(accum) / 128.0), 1.0);
}
*/
