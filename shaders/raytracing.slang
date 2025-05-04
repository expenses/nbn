import uniforms;
import shared.constants;
import shared.barycentrics;
import fallback_bindings;

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


bool trace_shadow_ray(
    UniformBuffer* uniforms,
    RayDesc desc
) {
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
