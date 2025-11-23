import shared.colour;
import shared.util;
import fallback_bindings;
import thief_models;
import shared.barycentrics;

struct PushConstants {
    float4x4 view_inv;
    float4x4 proj_inv;
    uint64_t as;
    Model* models;
    uint2 extent;
    uint image;
}

    [shader("compute")][NumThreads(8, 8, 1)] void
    write(uint2 thread : SV_DispatchThreadID, uniform PushConstants constants) {
    if (any(thread >= constants.extent)) {
        return;
    }

    let as = RaytracingAccelerationStructure(constants.as);

    let d = thread_to_clip(thread, constants.extent);
    let origin = mul(constants.view_inv, float4(0.0, 0.0, 0.0, 1.0)).xyz;
    let temp = mul(constants.proj_inv, float4(d.x, d.y, 1.0, 1.0));
    let direction =
        mul(constants.view_inv, float4(normalize(temp.xyz), 0.0)).xyz;

    RayDesc desc;
    desc.Direction = direction;
    desc.Origin = origin;
    desc.TMax = 10000.0;
    desc.TMin = 0.1;

    var query = RayQuery<RAY_FLAG_NONE>();
    query.TraceRayInline(
        as,
        RAY_FLAG_NONE,
        0xff,
        desc
    );

    query.Proceed();

    var value = float3(0.0);

    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        value = handle_hit(query, constants);
    }

    WTexture2D<float4> output = storage_images[constants.image];
    output.Store(thread, float4(srgb_lin2encoded(value), 1));
}

float3 handle_hit(RayQuery query, PushConstants constants) {
    let barycentrics = query.CommittedTriangleBarycentrics();
    let model_index = query.CommittedRayInstanceId();
    let primitive_index = query.CommittedRayPrimitiveIndex();
    let model = constants.models[model_index];

    let indices = uint3(
        model.read_index(primitive_index * 3 + 0),
        model.read_index(primitive_index * 3 + 1),
        model.read_index(primitive_index * 3 + 2),
    );

    let uv = interpolate_from_buffer(model.uvs, indices, barycentrics);
    let normal = interpolate_from_buffer(model.normals, indices, barycentrics);
    let position = interpolate_from_buffer(model.positions, indices, barycentrics);

    let l = normalize(float3(1,5,2));
    let f = abs(dot(normal,l));

    let col = colour_from_val(model.image) + (uv.xyy %1) * 0.1;
    return f *col;
}
