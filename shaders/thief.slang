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

    float3 get_unshadowed_contribution(float rand) {
        let light_index = uint(rand * uniforms.num_lights);
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

    UnshadowedContribution get_unshadowed_contribution(uint blue_noise_state) {
        var blue_noise = ThreadBlueNoise(uniforms.blue_noise_data, blue_noise_state);

        UnshadowedContribution contribution;
        contribution.contribution = get_unshadowed_contribution(blue_noise.rand());
        contribution.inv_pdf = uniforms.num_lights;
        return contribution;
    }

    float3 get_shadowed_contribution(float rand) {
        let light_index = uint(rand * uniforms.num_lights);
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

    float3 get_shadowed_contribution(uint blue_noise_state) {
        var blue_noise = ThreadBlueNoise(uniforms.blue_noise_data, blue_noise_state);
        return get_shadowed_contribution(blue_noise.rand());
    }
}

struct Environment: Contribution<uint> {
    SamplePoint point;
    Uniforms uniforms;
    float3 colour;

    UnshadowedContribution get_unshadowed_contribution(uint data) {
        UnshadowedContribution contribution;
        contribution.contribution = colour;
        contribution.inv_pdf = 1;
        return contribution;
    }

    float3 get_shadowed_contribution(float2 rand) {
        let hemisphere_dir = sample_cosine_weighted_hemisphere(rand);
        let rotated_dir = mul(build_tbn(point.normal), hemisphere_dir);

        RayDesc desc;
        desc.Origin = point.position;
        desc.Direction = rotated_dir;
        desc.TMin = 2e-4f;
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



        return visibility * colour;
    }

    float3 get_shadowed_contribution(uint blue_noise_state) {
        var blue_noise = ThreadBlueNoise(uniforms.blue_noise_data, blue_noise_state);
        return get_shadowed_contribution(blue_noise.rand2());
    }
}

struct Combined: Contribution<uint> {
    Environment env;
    State state;

    UnshadowedContribution get_unshadowed_contribution(uint blue_noise_state) {
        var blue_noise = ThreadBlueNoise(state.uniforms.blue_noise_data, blue_noise_state);
        let rand = blue_noise.rand2();

        UnshadowedContribution contribution;
        if (rand.x < 0.75) {
            contribution.contribution = state.get_unshadowed_contribution(rand.x/0.75);
            contribution.inv_pdf = state.uniforms.num_lights/0.75;
        } else {
            contribution.contribution = env.colour;
            contribution.inv_pdf = 1.0/0.25;
        }

        return contribution;
    }

    float3 get_shadowed_contribution(uint blue_noise_state) {
        var blue_noise = ThreadBlueNoise(state.uniforms.blue_noise_data, blue_noise_state);
        let rand = blue_noise.rand2();

        if (rand.x < 0.75) {
            return state.get_shadowed_contribution(rand.x/0.75);
        } else {
            return env.get_shadowed_contribution(float2(rand.x/0.25, rand.y));
        }
    }
}

float3 calculate_lighting_ris(uint2 thread, Uniforms uniforms, SamplePoint point) {
    var rng = LCG(thread, uniforms.frame_index);

    Combined state;
    state.state =  {point,uniforms};
    state.env = {point,uniforms,float3(0.1)};

    let reservoir_index = uniforms.thread_to_index(thread);

    var frame_reservoir = Reservoir<uint>(MAX_UINT_VALUE);

    uint num_samples = 8;
    for (uint i = 0; i < num_samples; i++) {
        let rng_state = pack_blue_noise_state(thread, uniforms.frame_index * num_samples + i);
        // Get the unshadowed contribution for a random light
        let contribution = state.get_unshadowed_contribution(rng_state);
        let p_hat = length(contribution.contribution);
        let weight = p_hat * contribution.inv_pdf;
        frame_reservoir.update(rng_state, weight, 1, rng.rand());
    }

    var contribution = frame_reservoir.data != MAX_UINT_VALUE ? state.get_shadowed_contribution(frame_reservoir.data) : float3(0.0);
    var p_hat = length(contribution);
    frame_reservoir.set_weight(p_hat);

    if (uniforms.flags & FrameFlags::TemporalReuse) {
        var temporal_reservoir = Reservoir<uint>(MAX_UINT_VALUE);

        // Update from current reservoir
        temporal_reservoir.update_from_reservoir(frame_reservoir, p_hat, &rng);

        // Get clamp previous reservoir and clamp to the last 20 frames.
        var previous_reservoir = uniforms.reservoirs[reservoir_index];
        previous_reservoir.clamp(20 * num_samples);

        // Update from prevous reservoir
        let p_hat_prev = length(state.get_unshadowed_contribution(previous_reservoir.data).contribution);
        temporal_reservoir.update_from_reservoir(previous_reservoir, p_hat_prev, &rng);

        p_hat = temporal_reservoir.data != MAX_UINT_VALUE ? length(state.get_unshadowed_contribution(temporal_reservoir.data).contribution) : 0.0;
        temporal_reservoir.set_weight(p_hat);
        frame_reservoir = temporal_reservoir;

        if (frame_reservoir.data != MAX_UINT_VALUE) {
            if (uniforms.flags & FrameFlags::OneRayPerPixel) {
                contribution = state.get_unshadowed_contribution(frame_reservoir.data).contribution;
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
