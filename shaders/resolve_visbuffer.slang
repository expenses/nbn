[vk::binding(0, 0)] __DynamicResource sampled_images[];
[vk::binding(1, 0)] __DynamicResource storage_images[];

import shared.constants;
import shared.util;
import shared.colour;
import shared.barycentrics;
import shared.derivatives;
import shared.sampling;
import models;
import uniforms;

float4 sample(uint index, WithDerivatives<float2> uv) {
    Sampler2D image = sampled_images[NonUniformResourceIndex(index)];
    return image.SampleGrad(uv.val, uv.dx, uv.dy);
}

float3 resolve_visbuffer_inner(uint2 thread, UniformBuffer* uniforms) {
    Texture2D<uint> input_image = storage_images[uniforms.visbuffer];

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

    let pos_a = model.positions[indices.x] + instance.position.xyz;
    let pos_b = model.positions[indices.y] + instance.position.xyz;
    let pos_c = model.positions[indices.z] + instance.position.xyz;

    let ndc = float2(thread) / float2(uniforms.extent) * 2.0 - 1.0;

    let bary = CalcFullBary(
        mul(uniforms.mat, float4(pos_a, 1.0)),
        mul(uniforms.mat, float4(pos_b, 1.0)),
        mul(uniforms.mat, float4(pos_c, 1.0)),
        ndc,
        uniforms.extent
    );

    let pos = interpolate(bary, pos_a, pos_b, pos_c);

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

    var base_colour = float3(1);
    var roughness = 1.0;
    var is_metallic = 0.0;

    if (material.normal_image != MAX_UINT_VALUE) {
        let sampled_normal = sample(material.normal_image, uv).xyz;

        if (all(sampled_normal != float3(0.0))) {
            let map_normal = sampled_normal * 255.0 / 127.0 - 128.0 / 127.0;
            normal = normalize(
                mul(compute_cotangent_frame(normal, pos, uv), map_normal)
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

    var factor = dot(sun, normal);
    if (factor > 0.0) {
        factor *= float(trace_shadow_ray(
            uniforms,
            pos.val,
            SphericalDirectionalLightRayDirection(
                interleaved_gradient_noise(thread.x, thread.y),
                sun,
                0.025
            )
        ));
    }

    factor = max(factor, .1);

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

    let sample = input_image
        [thread]; //.SampleLevel(thread_to_uv(thread, uniforms.extent), 0);

    let value = tony_mc_mapface(lut, sample);

    output_image.Store(thread, float4(srgb_lin2encoded(sample), 1.0));
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

bool trace_shadow_ray(
    UniformBuffer* uniforms,
    float3 position,
    float3 sun_direction
) {
    RayDesc desc;
    desc.Direction = sun_direction;
    desc.Origin = position;
    desc.TMax = 1e6f;
    desc.TMin = 0.1f;

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
    return query.CommittedStatus() != COMMITTED_TRIANGLE_HIT;
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
