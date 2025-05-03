import uniforms;
import shared.util;
import shared.constants;
import shared.barycentrics;
import shared.sampling;
import shared.noise;
import shared.subops;

[vk::binding(0, 0)] __DynamicResource sampled_images[];
[vk::binding(1, 0)] __DynamicResource storage_images[];

[shader("compute")][NumThreads(8, 8, 1)] void trace_shadow_rays(
    uint2 thread : SV_DispatchThreadID,
    uint2 local_thread : SV_GroupThreadID,
    uint2 group_id : SV_GroupID,
    //uint group_index: SV_GroupIndex,
    uniform UniformBuffer* uniforms
) {
    if (any(thread > uniforms.half_extent)) {
        return;
    }

    let uv = thread_to_uv(thread, uniforms.half_extent);

    Sampler2D<float> depthbuffer = sampled_images[uniforms.depthbuffer];

    let depth = depthbuffer.SampleLevel(uv, 0);

    let world_space_pos = world_position_from_depth(
        thread_to_clip(thread, uniforms.half_extent),
        depth,
        uniforms.mat_inv
    );

    let sun = normalize(float3(1.0, 2.0, 0.75));

    let in_shadow = trace_shadow_ray(
            uniforms,
            world_space_pos.xyz,
            SphericalDirectionalLightRayDirection(
                uniforms.blue_noise.sample(thread, uniforms.frame_index),
                sun,
                0.01
            )
        );

    let packed = wave64_pack_bools(in_shadow);

    if (WaveIsFirstLane()) {
        let index = group_id.x +  group_id.y * uniforms.half_extent.x/ 8;
        uniforms.half_size_shadow_buffer[index] = packed;    
    }
}

bool trace_shadow_ray(
    UniformBuffer* uniforms,
    float3 position,
    float3 sun_direction
) {
    RayDesc desc;
    desc.Direction = sun_direction;
    desc.Origin = position;
    desc.TMax = 1e6f;
    desc.TMin = 20.0;

    var query = RayQuery<
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
        | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES>();
    query.TraceRayInline(
        uniforms.acceleration_structure,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
            | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES,
        0xff,
        desc
    );
    while (query.Proceed()) {
        let base_colour = base_colour_from_ray(
            uniforms,
            query.CandidateTriangleBarycentrics(),
            query.CandidateRayInstanceCustomIndex(),
            query.CandidatePrimitiveIndex()
        );

        if (base_colour.a >= .5) {
            query.CommitNonOpaqueTriangleHit();
        }
    }
    return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}

float4 base_colour_from_ray(
    UniformBuffer* uniforms,
    float2 barycentrics,
    uint model_index,
    uint primitive_index,
) {
    let model = uniforms.models[model_index];

    if (model.material.base_colour_image == MAX_UINT_VALUE) {
        return float4(1.0);
    }

    let indices = uint3(
        model.read_index(primitive_index * 3 + 0),
        model.read_index(primitive_index * 3 + 1),
        model.read_index(primitive_index * 3 + 2),
    );

    let uv = interpolate_ray_barycentrics(
        model.uvs[indices.x],
        model.uvs[indices.y],
        model.uvs[indices.z],
        barycentrics
    );

    Sampler2D image =
        sampled_images[NonUniformResourceIndex(model.material.base_colour_image
        )];
    return image.SampleLevel(uv, 0);
}
