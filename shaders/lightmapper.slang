import lightmapper_structs;
import shared.noise;
import shared.sampling;
import shared.util;

struct Interp {
    float4 uv2_pos: SV_Position;
    float3 pos;
    float3 shading_normal;
};

[shader("vertex")] Interp vertex(uint vertex_index: SV_VertexID, uniform PushConstants push_constants) {
    let index = push_constants.model.read_index(vertex_index);
    let uv2 = push_constants.model.uv2s[index];

    Interp interp;
    interp.uv2_pos = float4(uv2 * 2.0 - 1.0, 0.0, 1.0);
    interp.pos = push_constants.model.positions[index];
    interp.shading_normal = push_constants.model.normals[index];
    return interp;
}

struct FragOutput {
    float4 pos: COLOR0;
    float4 shading_normal: COLOR1;
}

[shader("pixel")] FragOutput fragment(Interp in, uniform PushConstants push_constants) {
    FragOutput output;
    output.pos = float4(in.pos, 1.0);
    output.shading_normal = float4(in.shading_normal, 1.0);
    return output;
}

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
