import lightmapper_structs;
import fallback_bindings;

import shared.noise;
import shared.sampling;
import shared.util;
import shared.colour;
import shared.constants;
import shared.barycentrics;
import shared.envmap;
import shared.importance_sampling;

bool is_blocked(float3 position, float3 direction, float distance, PushConstants push_constants) {
    RayDesc desc;
    desc.Origin = position;
    desc.Direction = direction;
    desc.TMin = 0.0;
    desc.TMax = distance;
    var query = RayQuery<
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH>();
    query.TraceRayInline(
        push_constants.tlas,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        0xff,
        desc
    );
    query.Proceed();

    return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}

struct Sample {
    float3 radiance = float3(0);
    float3 direction = float3(0);

    float3 calculate(float3 normal) {
        return radiance * max(dot(normal, direction), 0.0);
    }

    float3 x() {
        return radiance * direction.x;
    }

    float3 y() {
        return radiance * direction.y;
    }

    float3 z() {
        return radiance * direction.z;
    }
}

static const float BOUNCE_PDF = PI;

Sample get_direct_point_lighting(
    float3 position,
    Optional<float3> normal,
    float2 rand,
    PushConstants push_constants
) {
    rand.x *= push_constants.num_lights;
    let light_index = uint(rand.x);
    rand.x = frac(rand.x);
    let light = push_constants.lights[light_index];

    var emission = light.emission;

    let vector = (light.position + sample_sphere(rand) * 0.25) - position;
    let distance_sq = dot(vector,vector);
    let distance = sqrt(distance_sq);

    Sample sample;
    sample.radiance = float3(0);
    sample.direction = vector / distance;

    if (let normal = normal) {
        if (dot(normal, sample.direction) <= 0) {
            return sample;
        }
    }

    // spotlights
    {
        let theta = dot(sample.direction, light.spotlight_direction);
        emission *= saturate(theta * light.spotlight_angle_scale + light.spotlight_angle_offset);
    }

    if (!is_blocked(position, sample.direction, distance, push_constants)) {
        sample.radiance = push_constants.num_lights * emission / distance_sq;
    }

    return sample;
}

Sample get_direct_env_map_lighting(
    float3 position,
    Optional<float3> normal,
    float2 rand,
    PushConstants push_constants
) {
    let envmap_sample = sample_envmap_by_rand(push_constants.alias_table, sampled_images[push_constants.envmap], push_constants.envmap_size, rand);

    Sample sample;
    sample.radiance = float3(0);
    sample.direction = envmap_sample.direction;

    var mis_weight = 1.0;

    if (let normal = normal) {
        if (dot(normal, sample.direction) <= 0) {
            return sample;
        }

        mis_weight = mis_power_heuristic(envmap_sample.pdf, BOUNCE_PDF);
    }

    if (!is_blocked(position, sample.direction, 200, push_constants)) {
        sample.radiance = envmap_sample.radiance / envmap_sample.pdf * mis_weight;
    }

    return sample;
}

Sample get_indirect_lighting(
    float3 position,
    Optional<float3> normal,
    float3 direction,
    PushConstants push_constants
) {
    Sample sample;
    sample.direction = direction;
    sample.radiance = float3(0);

    if (!(is_blocked(position, direction, 200, push_constants))) {
        let envmap_sample = sample_envmap_by_direction(push_constants.alias_table, sampled_images[push_constants.envmap], push_constants.envmap_size, direction);
        let bounce_pdf = BOUNCE_PDF;
        let mis_weight = mis_power_heuristic(bounce_pdf, envmap_sample.pdf);
        sample.radiance = envmap_sample.radiance / bounce_pdf * mis_weight;
    }

    return sample;
}

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

    var accum = float3(0);

    var rng = SobolRng(thread, push_constants.sample_index);

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

    //accum += get_direct_point_lighting(info.position, info.normal, rng.rand2(), push_constants).calculate(info.normal);
    accum += get_direct_env_map_lighting(info.position, info.normal, rng.rand2(), push_constants).calculate(info.normal);

    let bounce_direction = mul(build_tbn(info.normal), sample_uniform_hemisphere(&rng));
    accum += get_indirect_lighting(info.position, info.normal, bounce_direction, push_constants).calculate(info.normal);

    push_constants.temp[loc].rgb += accum;
    push_constants.temp[loc].a = 1.0;
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
        let s = sample_envmap_by_direction(constants.base.alias_table, sampled_images[constants.base.envmap], constants.base.envmap_size, direction);
        value = s.radiance;
    }

    WTexture2D<float4> output = storage_images[constants.image];
    output.Store(thread, float4(srgb_lin2encoded(value), 1));
}
