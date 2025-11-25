import shared.colour;
import shared.util;
import fallback_bindings;
import thief_structs;
import shared.barycentrics;
import shared.restir;
import shared.constants;

struct PrimInterp {
    float4 pos: SV_Position;
    uint prim;
};

[shader("vertex")]
PrimInterp mesh_vert(uint vertex_id : SV_VertexID, uniform PushConstants constants) {
    let uniforms = constants.uniforms;
    let model = uniforms.model;

    let index = model.indices[vertex_id];
    let position = model.positions[index];

    PrimInterp output;
    output.prim = vertex_id / 3;
    output.pos = mul(uniforms.proj_view, float4(position, 1.0));
    return output;
}

[shader("pixel")]
uint mesh_frag(PrimInterp interp) {
    return interp.prim;
}

struct SamplePoint {
    float3 position;
    float3 normal;
}

float3 get_unshadowed_contribution(SamplePoint point, uint light_index, Uniforms uniforms) {
    let light = uniforms.get_light(light_index);

    let vector = point.position - light.position;
    let distance_sq = dot(vector,vector);
    let distance = sqrt(distance_sq);

    let direction = vector / distance;

    return light.color * (light.intensity * max(dot(point.normal, direction),0.0) / distance_sq);
}

float3 get_shadowed_contribution(SamplePoint point, uint light_index, Uniforms uniforms) {
    let light = uniforms.get_light(light_index);

    let vector = point.position - light.position;
    let distance_sq = dot(vector,vector);
    let distance = sqrt(distance_sq);

    let direction = vector / distance;

    RayDesc desc;
    desc.Origin = point.position;
    desc.Direction = -direction;
    desc.TMin = 1e-3f;
    desc.TMax = distance;
    var query = RayQuery<
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH>();
    query.TraceRayInline(
        RaytracingAccelerationStructure(uniforms.tlas),
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        0xff,
        desc
    );
    query.Proceed();
    let visibility = float(query.CommittedStatus() != COMMITTED_TRIANGLE_HIT);

    return light.color * (visibility * light.intensity * max(dot(point.normal, direction),0.0) / distance_sq);
}

float3 calculate_lighting_ris(uint2 thread, Uniforms uniforms, SamplePoint point) {
    var blue_noise = uniforms.blue_noise(thread);

    let reservoir_index = uniforms.thread_to_index(thread);
    let pdf = rcp(uniforms.num_lights);

    var state = RestirState<int>(pdf, -1);

    uint num_samples = 8;
    for (uint i = 0; i < num_samples; i++) {
        uint light_index = uint(blue_noise.rand() * uniforms.num_lights);
        let p_hat = length(get_unshadowed_contribution(point, light_index, uniforms));
        state.update(light_index, p_hat, &blue_noise);
    }

    var contribution = state.get_data() != -1 ? get_shadowed_contribution(point, state.get_data(), uniforms) : float3(0.0);
    var p_hat = length(contribution);
    state.update_from_frame_sample(p_hat);

    if (uniforms.temporal_reuse!=0) {
        var temporal_reservoir = Reservoir<int>(-1);

        // Update from current reservoir
        temporal_reservoir.update_from_reservoir(state.res, p_hat, &blue_noise);

        // Get clamp previous reservoir and clamp
        var previous_reservoir = uniforms.reservoirs[reservoir_index];
        previous_reservoir.M = min(20.0 * state.res.M, previous_reservoir.M);

        // Update from prevous reservoir
        let p_hat_prev = length(get_unshadowed_contribution(point, previous_reservoir.get_data(), uniforms));
        temporal_reservoir.update_from_reservoir(previous_reservoir, p_hat_prev, &blue_noise);

        let p_hat = temporal_reservoir.get_data() != -1 ? length(get_unshadowed_contribution(point, temporal_reservoir.get_data(), uniforms)) : 0.0;
        temporal_reservoir.set_weight(p_hat);
        state.res = temporal_reservoir;
    }

    contribution = state.get_data() != -1 ? get_shadowed_contribution(point, state.get_data(), uniforms) : float3(0.0);
    p_hat = length(contribution);
    state.update_from_frame_sample(p_hat);


    uniforms.reservoirs[reservoir_index] = state.res;

    return contribution * state.get_weight();
}

struct LightInterp {
    float4 pos: SV_Position;
    float3 col;
    float point_size: SV_PointSize;
};

[shader("vertex")] LightInterp lights_vert(uint vertex_id : SV_VertexID, uniform PushConstants constants) {
    let uniforms = constants.uniforms;
    let light = uniforms.get_light(vertex_id);

    LightInterp output;
    output.col = light.color * light.intensity;
    output.pos = mul(uniforms.proj_view, float4(light.position, 1.0));
    output.point_size = 2.0;
    return output;
}

[shader("pixel")] float3 lights_frag(LightInterp interp) {
    return interp.col;
}

[shader("compute")][NumThreads(8, 8, 1)] void resolve(uint2 thread : SV_DispatchThreadID, uniform PushConstants constants) {
    Uniforms uniforms = *constants.uniforms;

    if (any(thread >= uniforms.extent)) {
        return;
    }

    Texture2D<uint> prim_image = storage_images[uniforms.prims_image];
    let prim = prim_image[thread];
    let value = resolve_inner(thread, uniforms, prim);

    RWTexture2D<float3> hdr = storage_images[uniforms.hdr_image];
    hdr[thread] = lerp(hdr[thread], value, 1.0/float(uniforms.accum_index+1));

    WTexture2D<float4> output = storage_images[uniforms.swapchain_image];
    output.Store(thread, float4(srgb_lin2encoded(hdr[thread]), 1));
}

float3 resolve_inner(uint2 thread, Uniforms uniforms, uint prim) {
    if (prim == MAX_UINT_VALUE) {
        return float3(0);
    }

    let model = uniforms.model;

    let indices = uint3(
        model.read_index(prim * 3 + 0),
        model.read_index(prim * 3 + 1),
        model.read_index(prim * 3 + 2),
    );

    let bary_and_pos = calculate_barycentrics_from_world_positions(
        uniforms.proj_view,
        model.positions[indices.x],
        model.positions[indices.y],
        model.positions[indices.z],
        thread,
        uniforms.extent
    );
    let pos = bary_and_pos.position.val;
    let bary = bary_and_pos.barycentrics;

    let uv = interpolate_from_buffer(model.uvs, indices, bary);

    let normal = model.normals[indices.x];
    let image_index = model.image_indices[prim];

    Sampler2D<float3> img = sampled_images[NonUniformResourceIndex(image_index)];

    let col = img.SampleGrad(uv.val, uv.dx, uv.dy);

    SamplePoint point;
    point.position = pos;
    point.normal = normal;

    return col * calculate_lighting_ris(thread, uniforms, point);
}
