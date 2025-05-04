import uniforms;
import shared.constants;
import shared.barycentrics;
import fallback_bindings;
import shared.util;
import shared.derivatives;

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

bool trace_shadow_ray(UniformBuffer* uniforms, RayDesc desc) {
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

float4 sample(uint index, WithDerivatives<float2> uv) {
    Sampler2D image = sampled_images[NonUniformResourceIndex(index)];
    return image.SampleGrad(uv.val, uv.dx, uv.dy);
}

float3 trace_primary_ray(
    UniformBuffer* uniforms,
    float4x4 view_inv,
    float4x4 proj_inv,
    uint2 thread
) {
    let clip = thread_to_clip(thread, uniforms.extent);
    RayDesc desc;
    desc.Origin = mul(view_inv, float4(0.0, 0.0, 0.0, 1.0)).xyz;
    let temp = mul(proj_inv, float4(clip, 1.0, 1.0));
    desc.Direction = mul(view_inv, float4(normalize(temp.xyz), 0.0)).xyz;
    desc.TMax = 1e6f;
    desc.TMin = 20.0;
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
    if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
        return float3(1, 0, 0);
    }

    let primitive_index = query.CommittedPrimitiveIndex();
    let model_index = query.CommittedRayInstanceCustomIndex();
    let model = uniforms.models[model_index];
    let material = model.material;

    if (model.material.base_colour_image == MAX_UINT_VALUE) {
        return float3(1.0);
    }
    let indices = uint3(
        model.read_index(primitive_index * 3 + 0),
        model.read_index(primitive_index * 3 + 1),
        model.read_index(primitive_index * 3 + 2),
    );
    let bary_and_pos = calculate_barycentrics_from_world_positions(
        uniforms.mat,
        model.positions[indices.x], // + instance.position.xyz,
        model.positions[indices.y], // + instance.position.xyz,
        model.positions[indices.z], // + instance.position.xyz,
        thread,
        uniforms.extent
    );
    let uv = interpolate(
        bary_and_pos.barycentrics,
        model.uvs[indices.x],
        model.uvs[indices.y],
        model.uvs[indices.z]
    );
    return sample(material.base_colour_image, uv).rgb;
}