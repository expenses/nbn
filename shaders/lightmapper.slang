import lightmapper_structs;
import fallback_bindings;

import shared.noise;
import shared.sampling;
import shared.util;
import shared.colour;
import shared.constants;
import shared.barycentrics;
import shared.sobol;
import shared.restir;

float3 ris<C: Contribution<float2>>(C contributor, uint num_samples, SobolRng* rng) {
    var reservoir = Reservoir<float2>(float2(-1));
    var lcg = LCG(rng.thread, rng.sample);

    for (uint i = 0; i < num_samples; i++) {
        let rng_state = rng.rand2();
        // Get the unshadowed contribution for a random light
        let value = contributor.unshadowed_contribution(rng_state);
        reservoir.update(rng_state, value.weight(), 1, lcg.rand());
    }

    let value = any(reservoir.data != float2(-1)) ? contributor.shadowed_contribution(reservoir.data) : ShadowedValue();
    reservoir.set_weight_from_p(value.p);
    return value.contribution * reservoir.weight;
}

struct ContributionInfo {
    UnshadowedValue unshadowed;
    float3 direction;
    float distance;
    float3 contribution;
}

struct Lights: Contribution<float2> {
    HitInfo info;
    PushConstants push_constants;

    ContributionInfo contribution_info(float2 rand) {
        rand.x *= push_constants.num_lights;
        let light_index = uint(rand.x);
        rand.x = frac(rand.x);
        let light = push_constants.lights[light_index];

        var emission = light.emission;

        let vector = (light.position + sample_shere(rand) * 0.25) - info.position;
        let distance_sq = dot(vector,vector);
        let distance = sqrt(distance_sq);

        let direction = vector / distance;

        // spotlights
        {
            let theta = dot(direction, light.spotlight_direction);
            emission *= saturate(theta * light.spotlight_angle_scale + light.spotlight_angle_offset);
        }

        let n_dot_l = max(dot(info.normal, direction), 0.0);

        ContributionInfo contribution;
        contribution.contribution = emission * (n_dot_l / distance_sq);
        contribution.unshadowed.p_hat = length(contribution.contribution);
        contribution.unshadowed.inv_pdf = float(push_constants.num_lights);
        contribution.distance = distance;
        contribution.direction = direction;
        return contribution;
    }

    UnshadowedValue unshadowed_contribution(float2 rand) {
        return contribution_info(rand).unshadowed;
    }

    ShadowedValue shadowed_contribution(float2 rand) {
        var contribution_info = contribution_info(rand);
        RayDesc desc;
        desc.Origin = info.position;
        desc.Direction = contribution_info.direction;
        desc.TMin = 0.0;
        desc.TMax = contribution_info.distance;
        var query = RayQuery<
            RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH>();
        query.TraceRayInline(
            push_constants.tlas,
            RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
            0xff,
            desc
        );
        query.Proceed();

        ShadowedValue value;
        value.p = contribution_info.unshadowed.p_hat;
        value.contribution = contribution_info.contribution;

        if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            value.contribution = float3(0);
            value.p = 0.0;
        }
        return value;
    }
}

struct EnvMap: Contribution<float2> {
    PushConstants push_constants;
    HitInfo info;

    ContributionInfo contribution_info(float2 rand2) {
        var index = uint(rand2.x * push_constants.envmap_size.x * push_constants.envmap_size.y);
        var alias = push_constants.alias_table[index];
        if (rand2.y > alias.threshold) {
            index = alias.fallback;
            alias = push_constants.alias_table[alias.fallback];
        }
        let uv = (float2(index % push_constants.envmap_size.x, index / push_constants.envmap_size.x) + 0.5) / float2(push_constants.envmap_size);
        Sampler2D envmap = sampled_images[push_constants.envmap];

        let rotated_dir = uv_to_direction(uv);

        let n_dot_l = max(dot(info.normal, rotated_dir), 0.0);

        ContributionInfo contribution;
        contribution.contribution = envmap.SampleLevel(uv, 0).rgb * n_dot_l;
        contribution.unshadowed.p_hat = length(contribution.contribution);
        contribution.unshadowed.inv_pdf = alias.inv_pdf;
        contribution.distance = 200;
        contribution.direction = rotated_dir;
        return contribution;
    }

    UnshadowedValue unshadowed_contribution(float2 rand2) {
        return contribution_info(rand2).unshadowed;
    }

    ShadowedValue shadowed_contribution(float2 rand2) {
        var index = uint(rand2.x * push_constants.envmap_size.x * push_constants.envmap_size.y);
        var alias = push_constants.alias_table[index];
        if (rand2.y > alias.threshold) {
            index = alias.fallback;
            alias = push_constants.alias_table[alias.fallback];
        }
        let uv = (float2(index % push_constants.envmap_size.x, index / push_constants.envmap_size.x) + 0.5) / float2(push_constants.envmap_size);
        Sampler2D envmap = sampled_images[push_constants.envmap];

        let rotated_dir = uv_to_direction(uv);

        let n_dot_l = max(dot(info.normal, rotated_dir), 0.0);

        ShadowedValue contribution;
        contribution.contribution = float3(0);
        contribution.p = 0.0;

        if (n_dot_l == 0.0) {
            return contribution;
        }

        RayDesc desc;
        desc.Direction = rotated_dir;
        desc.Origin = info.position;
        desc.TMax = 200.0;
        desc.TMin = 0;

        var query = RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH>();
        query.TraceRayInline(
            push_constants.tlas,
            RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
            0xff,
            desc
        );
        query.Proceed();

        if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
            contribution.contribution = envmap.SampleLevel(uv, 0).rgb * n_dot_l;
            contribution.p = length(contribution.contribution);
        }

        return contribution;
    }
}

float2 envmap_uv(float3 direction) {
    return float2(
        atan2(direction.z, direction.x) / (2.0 * PI),
        acos(direction.y) / PI
    );
}

float3 uv_to_direction(float2 uv) {
    let phi   = uv.x * 2.0 * PI;
    let theta = uv.y * PI;
    return float3(
        sin(theta) * cos(phi),
        cos(theta),
        sin(theta) * sin(phi),
    );
}

struct SobolRng {
    uint pixel_seed = 0;
    uint sample = 0;
    uint dimension = 0;
    uint2 thread;

    __init(uint2 thread_, uint sample_) {
        thread = thread_;
        pixel_seed = hp_mix32(thread.x ^ (thread.y << 16));
        sample = sample_;
    }

    [mutating]
    float rand() {
        let rand = sobol_burley_sample_1D(sample, dimension, pixel_seed);
        dimension++;
        return rand;
    }

    [mutating]
    float2 rand2() {
        let rand = sobol_burley_sample_2D(sample, dimension, pixel_seed);
        dimension++;
        return rand;
    }

    [mutating]
    float3 rand3() {
        let rand = sobol_burley_sample_3D(sample, dimension, pixel_seed);
        dimension++;
        return rand;
    }
}

[shader("compute")]
[NumThreads(8,8,1)]
void lightmap(uint2 thread : SV_DispatchThreadID, uniform PushConstants push_constants) {
    if (any(thread >= push_constants.extent)) {
        return;
    }

    for (uint vv = 0; vv < 8; vv++) {
        let loc = thread.y * push_constants.extent.x + thread.x;
        let location_bitmask = push_constants.location_bitmasks[loc];

        if (location_bitmask == 0) {
            return;
        }

        var accum = float3(0);

        var rng = SobolRng(thread, push_constants.sample_index * 4 + vv);

        let bit = random_bit(location_bitmask, rng.rand());
        let triangle_index = push_constants.triangle_indices[loc * 64 + bit];
        let indices = push_constants.model.read_indices(triangle_index);

        let uv_ray_offset = frac(0.5 + R2(bit));

        let uv = (float2(thread)+uv_ray_offset)/push_constants.extent;

        let barycentrics = barycentrics2d(
            uv,
            push_constants.model.uv2s[indices.x],
            push_constants.model.uv2s[indices.y],
            push_constants.model.uv2s[indices.z]
        );

        let info = generate_hit(barycentrics, triangle_index, push_constants.model);

        accum += get_lighting(info, push_constants, &rng);

        //if (let infov = bounce(info, rng.rand2(), push_constants)) {
        //    accum += infov.diffuse * get_lighting(infov, push_constants, &rng);
        //}

        push_constants.temp[loc].rgb += accum;
        push_constants.temp[loc].a = 1.0;
    }
}

bool check_for_backfaces(float2 barycentrics, uint triangle_index, PushConstants push_constants) {
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
            push_constants.tlas,
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

    let loc = thread.y * push_constants.extent.x + thread.x;

    uint64_t bitmask = 0;

    for (uint i = 0; i < 64; i++) {
        let uv_ray_offset = frac(0.5 + R2(i));

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
            push_constants.uv_tlas,
            RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
            0xff,
            tri_desc
        );
        tri_query.Proceed();
        var hit = tri_query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
        if (hit) {
            hit = !check_for_backfaces(tri_query.CommittedRayBarycentrics(), tri_query.CommittedPrimitiveIndex(), push_constants);
            push_constants.triangle_indices[loc * 64 + i] = tri_query.CommittedPrimitiveIndex();
        }
        bitmask |= uint64_t(hit) << i;
    }

    push_constants.location_bitmasks[loc] = bitmask;
}

struct PartitionedRand {
    uint index;
    float rand;
    float inv_probability;
};

PartitionedRand partition_rand(float rand, uint num_values) {
    let scaled = rand * num_values;
    PartitionedRand part;
    part.index = uint(scaled);
    part.rand = frac(scaled);
    part.inv_probability = float(num_values);
    return part;
}

struct HitInfo {
    float3 position;
    float3 normal;
    float3 diffuse;
};

HitInfo generate_hit(float2 barycentrics, uint triangle_index, Model* model) {
    let indices = model.read_indices(triangle_index);
    let a = model.positions[indices.x];
    let b = model.positions[indices.y];
    let c = model.positions[indices.z];

    let geom_normal = normalize(cross(b - a, c - a));

    HitInfo info;
    info.position = offset_ray(shadow_terminator_hack(
        a, b, c,
        model.normals[indices.x],
        model.normals[indices.y],
        model.normals[indices.z],
        float3(1.0-barycentrics.x-barycentrics.y, barycentrics)
    ), geom_normal);

    info.normal = normalize(interpolate_from_buffer(model.normals, indices, barycentrics));

    let material_index = model.material_indices[triangle_index];
    let material = model.materials[material_index];

    info.diffuse = material.base_color_factor.rgb;

    if (material.base_colour_image != MAX_UINT_VALUE) {
        let uv = interpolate_from_buffer(model.uvs, indices, barycentrics);
        Sampler2D img = sampled_images[NonUniformResourceIndex(material.base_colour_image)];
        info.diffuse *= img.SampleLevel(uv, 0).rgb;
    }

    return info;
}

Optional<HitInfo> bounce(HitInfo info, float2 rand2, PushConstants push_constants) {
    RayDesc desc;
    desc.Origin = info.position;
    desc.Direction = mul(build_tbn(info.normal), sample_cosine_weighted_hemisphere(rand2));
    desc.TMax = 200.0;
    desc.TMin = 0;

    var query = RayQuery<RAY_FLAG_NONE>();
    query.TraceRayInline(
        push_constants.tlas,
        RAY_FLAG_NONE,
        0xff,
        desc
    );
    query.Proceed();

    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        return generate_hit(query.CommittedRayBarycentrics(), query.CommittedPrimitiveIndex(), push_constants.model);
    }

    return none;
}

float3 get_lighting(HitInfo info, PushConstants push_constants, SobolRng* rng) {
    Lights lights;
    lights.info = info;
    lights.push_constants = push_constants;

    EnvMap env_map;
    env_map.info = info;
    env_map.push_constants = push_constants;

    return ris(lights, 1, rng);//lights.shadowed_contribution2(rng.rand2()).adjusted();// + env_map.shadowed_contribution2(rng.rand2()).adjusted();
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
    if (any(thread >= constants.base.extent)) {
        return;
    }

    let model = constants.base.model;

    let d = thread_to_clip(thread, constants.base.extent);
    let origin = mul(constants.view_inv, float4(0.0, 0.0, 0.0, 1.0)).xyz;
    let temp = mul(constants.proj_inv, float4(d.x, d.y, 1.0, 1.0));
    let direction =
        mul(constants.view_inv, float4(normalize(temp.xyz), 0.0)).xyz;

    RayDesc desc;
    desc.Direction = direction;
    desc.Origin = origin;
    desc.TMax = 200.0;
    desc.TMin = 0.0;

    var query = RayQuery<RAY_FLAG_NONE>();
    query.TraceRayInline(
        constants.base.tlas,
        RAY_FLAG_NONE,
        0xff,
        desc
    );
    query.Proceed();

    var value = float3(0.0);

    var rng = SobolRng(thread, constants.frame_index);

    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        let hit = generate_hit(query.CommittedTriangleBarycentrics(), query.CommittedPrimitiveIndex(), constants.base.model);

        value = hit.diffuse;
    } else {
        Sampler2D image = sampled_images[constants.base.envmap];
        value = image.SampleLevel(envmap_uv(direction), 0).rgb;
    }

    WTexture2D<float4> output = storage_images[constants.image];
    output.Store(thread, float4(srgb_lin2encoded(value), 1));
}
