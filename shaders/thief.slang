import shared.colour;
import shared.util;
import fallback_bindings;
import thief_structs;
import shared.barycentrics;
import shared.restir;
import shared.constants;
import shared.noise;
import shared.sampling;

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

struct State: Contribution<uint> {
    SamplePoint point;
    Uniforms uniforms;

    float3 get_unshadowed_contribution(uint blue_noise_state) {
        var blue_noise = ThreadBlueNoise(uniforms.blue_noise_data, blue_noise_state);
        let light_index = uint(blue_noise.rand() * uniforms.num_lights);
        let light = uniforms.get_light(light_index);

        var emission = light.emission;

        let vector = light.position - point.position;
        let distance_sq = dot(vector,vector);
        let distance = sqrt(distance_sq);

        let direction = vector / distance;

        // spotlights
        {
            let theta = dot(direction, light.spotlight_direction);
            emission *= saturate(theta * light.spotlight_angle_scale + light.spotlight_angle_offset);
        }

        return emission * (max(dot(point.normal, direction), 0.0) / distance_sq);
    }

    float3 get_shadowed_contribution(uint blue_noise_state) {
        var blue_noise = ThreadBlueNoise(uniforms.blue_noise_data, blue_noise_state);
        let light_index = uint(blue_noise.rand() * uniforms.num_lights);
        let light = uniforms.get_light(light_index);

        var emission = light.emission;

        let vector = light.position - point.position;
        let distance_sq = dot(vector,vector);
        let distance = sqrt(distance_sq);

        let direction = vector / distance;

        // spotlights
        {
            let theta = dot(direction, light.spotlight_direction);
            emission *= saturate(theta * light.spotlight_angle_scale + light.spotlight_angle_offset);
        }

        RayDesc desc;
        desc.Origin = point.position;
        desc.Direction = direction;
        desc.TMin = 5e-3f;
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

        return emission * (visibility * max(dot(point.normal, direction),0.0) / distance_sq);
    }
}

struct Environment: Contribution<uint> {
    SamplePoint point;
    Uniforms uniforms;

    float3 get_unshadowed_contribution(uint data) {
        return float3(1);
    }

    float3 get_shadowed_contribution(uint blue_noise_state) {
        var blue_noise = ThreadBlueNoise(uniforms.blue_noise_data, blue_noise_state);

        let hemisphere_dir = sample_cosine_weighted_hemisphere(&blue_noise);
        let rotated_dir = mul(build_tbn(point.normal), hemisphere_dir);

        RayDesc desc;
        desc.Origin = point.position;
        desc.Direction = rotated_dir;
        desc.TMin = 1e-4f;
        desc.TMax = 100000.0;
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



        return visibility;
    }
}

float3 calculate_lighting_ris(uint2 thread, Uniforms uniforms, SamplePoint point) {
    var blue_noise = uniforms.blue_noise(thread);

    State state = {point,uniforms};// = {point,uniforms};

    let reservoir_index = uniforms.thread_to_index(thread);
    // the pdf is 1/num_lights but we're dividing by that
    let recip_pdf = uniforms.num_lights;

    var frame_reservoir = Reservoir<uint>(MAX_UINT_VALUE);

    uint num_samples = 8;
    for (uint i = 0; i < num_samples; i++) {
        // Sample the value for the reservoir update now, so that the
        // random values generated in `get_unshadowed_contribution` are different
        let update_rand = blue_noise.rand();

        // Get the unshadowed contribution for a random light
        let rng_state = blue_noise.state();
        let p_hat = length(state.get_unshadowed_contribution(rng_state));
        let weight = p_hat * recip_pdf;
        frame_reservoir.update(rng_state, weight, 1, update_rand);
    }

    var contribution = frame_reservoir.data != MAX_UINT_VALUE ? state.get_shadowed_contribution(frame_reservoir.data) : float3(0.0);
    var p_hat = length(contribution);
    frame_reservoir.set_weight(p_hat);

    if (uniforms.flags & FrameFlags::TemporalReuse) {
        var temporal_reservoir = Reservoir<uint>(MAX_UINT_VALUE);

        // Update from current reservoir
        temporal_reservoir.update_from_reservoir(frame_reservoir, p_hat, &blue_noise);

        // Get clamp previous reservoir and clamp to the last 20 frames.
        var previous_reservoir = uniforms.reservoirs[reservoir_index];
        previous_reservoir.clamp(20 * num_samples);

        // Update from prevous reservoir
        let p_hat_prev = length(state.get_unshadowed_contribution(previous_reservoir.data));
        temporal_reservoir.update_from_reservoir(previous_reservoir, p_hat_prev, &blue_noise);

        p_hat = temporal_reservoir.data != MAX_UINT_VALUE ? length(state.get_unshadowed_contribution(temporal_reservoir.data)) : 0.0;
        temporal_reservoir.set_weight(p_hat);
        frame_reservoir = temporal_reservoir;

        if (frame_reservoir.data != MAX_UINT_VALUE) {
            if (uniforms.flags & FrameFlags::OneRayPerPixel) {
                contribution = state.get_unshadowed_contribution(frame_reservoir.data);
            } else {
                contribution = state.get_shadowed_contribution(frame_reservoir.data);
            }
        } else {
            contribution = float3(0.0);
        }
        p_hat = length(contribution);
        frame_reservoir.set_weight(p_hat);
    }

    uniforms.reservoirs[reservoir_index] = frame_reservoir;

    return contribution * frame_reservoir.weight;
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
    output.col = light.emission;
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
    // normals are flipped
    point.normal = normal;

    if (dot(uniforms.camera_pos - pos, normal) < 0) {
        point.normal = -normal;
    }

    return col * calculate_lighting_ris(thread, uniforms, point);
}
