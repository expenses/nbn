import shared.constants;
import shared.util;
import shared.colour;
import shared.barycentrics;
import shared.derivatives;
import shared.linalg;
import shared.sampling;
import models;
import uniforms;
import raytracing;
import fallback_bindings;

float4 sample(uint index, WithDerivatives<float2> uv) {
    Sampler2D image = sampled_images[NonUniformResourceIndex(index)];
    return image.SampleGrad(uv.val, uv.dx, uv.dy);
}

float3 resolve_visbuffer_inner(uint2 thread, UniformBuffer* uniforms) {
    if (uniforms.debug_mode == DebugMode::RaytracedScene) {
            return trace_primary_ray(
                uniforms,
                inverse(uniforms.view),
                inverse(uniforms.perspective),
                thread
            );
            }

    Texture2D<uint> input_image = storage_images[uniforms.visbuffer];

    let shadow_thread = thread / 2;
    let shadow_buffer_index = (shadow_thread.x / 8)
        + (shadow_thread.y / 4) * div_ceil(uniforms.half_extent.x, 8);
    let packed_shadow_bits =
        uniforms.half_size_shadow_buffer[shadow_buffer_index];
    let packed_bit_index = (shadow_thread.x % 8) + (shadow_thread.y % 4) * 8;
    let is_shadows_val =
        float(!bool((packed_shadow_bits >> packed_bit_index) & 1));

    let val = input_image[thread];

    if (val == MAX_UINT_VALUE) {
        return float3(0.2, 0.2, 0.6);
    }

    let meshlet_instance_index = val >> 7;
    let triangle_index = val & 0b1111111;
    let meshlet_instance = uniforms.meshlet_instances[meshlet_instance_index];
    let meshlet = meshlet_instance.meshlet;
    let instance = meshlet_instance.instance;
    let model = uniforms.models[instance.model_index];
    let material = model.material;

    let indices = uint3(
        model.vertices
            [meshlet.vertex_offset
             + model.triangles
                   [meshlet.triangle_offset + triangle_index * 3 + 0]],
        model.vertices
            [meshlet.vertex_offset
             + model.triangles
                   [meshlet.triangle_offset + triangle_index * 3 + 1]],
        model.vertices
            [meshlet.vertex_offset
             + model.triangles
                   [meshlet.triangle_offset + triangle_index * 3 + 2]],
    );

    let bary_and_pos = calculate_barycentrics_from_world_positions(
        uniforms.mat,
        model.positions[indices.x] + instance.position.xyz,
        model.positions[indices.y] + instance.position.xyz,
        model.positions[indices.z] + instance.position.xyz,
        thread,
        uniforms.extent
    );
    let pos = bary_and_pos.position;
    let bary = bary_and_pos.barycentrics;

    let uv = interpolate(
        bary,
        model.uvs[indices.x],
        model.uvs[indices.y],
        model.uvs[indices.z]
    );

    var normal = normalize(interpolate(
                               bary,
                               model.normals[indices.x],
                               model.normals[indices.y],
                               model.normals[indices.z]
    )
                               .val);

    let base_normal = normal;
    var map_normal = normal;

    var base_colour = float3(1);
    var roughness = 1.0;
    var is_metallic = 0.0;

    if (material.normal_image != MAX_UINT_VALUE) {
        let sampled_normal = sample(material.normal_image, uv).xyz;

        if (all(sampled_normal != float3(0.0))) {
            map_normal = sampled_normal * 255.0 / 127.0 - 128.0 / 127.0;
            normal = normalize(
                mul(map_normal, compute_cotangent_frame(normal, pos, uv))
            );
        }
    }

    if (material.metallic_roughness_image != MAX_UINT_VALUE) {
        let metal_roughness = sample(material.metallic_roughness_image, uv);
        roughness = metal_roughness.g;
        is_metallic = metal_roughness.b;
    }

    if (material.base_colour_image != MAX_UINT_VALUE) {
        base_colour = sample(material.base_colour_image, uv).rgb;
    }

    var emission = float3(0);
    let emission_factor = 100.0;
    if (material.emissive_image != MAX_UINT_VALUE) {
        emission = sample(material.emissive_image, uv).rgb * emission_factor;
    }

    let sun = normalize(float3(1.0, 2.0, 0.75));

    var factor = dot(sun, normal) * is_shadows_val;

    factor = max(factor, .1);

    switch (uniforms.debug_mode) {
        case DebugMode::Triangles:
            return colour_from_val(triangle_index);
        case DebugMode::Model:
            return colour_from_val(instance.model_index);
        case DebugMode::BaseColour:
            return base_colour;
        case DebugMode::Normals:
            return (normal + 1.0) / 2.0;
        case DebugMode::BaseNormals:
            return (base_normal + 1.0) / 2.0;
        case DebugMode::MapNormals:
            return (map_normal + 1.0) / 2.0;
        case DebugMode::Metallic:
            return is_metallic;
        case DebugMode::Roughness:
            return roughness;
    }

    return (factor * base_colour) + emission;
}

[shader("compute")][NumThreads(8, 8, 1)] void resolve_visbuffer(
    uint2 thread : SV_DispatchThreadID,
    uniform UniformBuffer* uniforms
) {
    if (any(thread > uniforms.extent)) {
        return;
    }

    WTexture2D<float4> output_image = storage_images[uniforms.hdrbuffer];
    let output = resolve_visbuffer_inner(thread, uniforms);
    output_image.Store(thread, float4(output, 1.0));
}

    [shader("compute")][NumThreads(8, 8, 1)] void tonemap(
        uint2 thread : SV_DispatchThreadID,
        uniform UniformBuffer* uniforms
    ) {
    Sampler3D<float3> lut = sampled_images[uniforms.tonemap_lut_image];
    Sampler2D<float3> input_image = storage_images[uniforms.hdrbuffer];
    WTexture2D<float4> output_image = storage_images[uniforms.swapchain_image];

    if (any(thread > uniforms.extent)) {
        return;
    }

    // let sample = compute_dof(thread_to_uv(thread, uniforms.extent),
    //     sampled_images[uniforms.depthbuffer],
    //     input_image,
    //     uniforms.extent,
    //     uniforms.near_plane
    //     );

    var value = input_image
        [thread]; //.SampleLevel(thread_to_uv(thread, uniforms.extent), 0);

    if (uniforms.debug_mode == DebugMode::None) {
        value = tony_mc_mapface(lut, value);
    }

    output_image.Store(thread, float4(srgb_lin2encoded(value), 1.0));
}

Optional<float3> trace(UniformBuffer* uniforms, float3 position, float3 dir) {
    RayDesc desc;
    desc.Direction = dir;
    desc.Origin = position;
    desc.TMax = 1e6f;
    desc.TMin = 0.1f;

    var query = RayQuery<RAY_FLAG_NONE>();
    query.TraceRayInline(
        uniforms.acceleration_structure,
        RAY_FLAG_NONE,
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

    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        let base_colour = base_colour_from_ray(
            uniforms,
            query.CommittedTriangleBarycentrics(),
            query.CommittedRayInstanceCustomIndex(),
            query.CommittedPrimitiveIndex()
        );

        return base_colour.rgb;
    } else {
        return none;
    }
}
