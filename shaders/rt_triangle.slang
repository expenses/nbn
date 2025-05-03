import shared.colour;
import shared.util;

[vk::binding(1, 0)] __DynamicResource storage_images[];

struct PushConstants {
    float4x4 view_inv;
    float4x4 proj_inv;
    uint64_t as;
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
    desc.TMax = 200.0;
    desc.TMin = 0.1;

    var query = RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH>();
    query.TraceRayInline(
        as,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        0xff,
        desc
    );

    query.Proceed();

    var value = float3(0.0);

    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        let barycentrics = query.CommittedTriangleBarycentrics();
        value = float3(1.0 - barycentrics.x - barycentrics.y, barycentrics);
    }

    WTexture2D<float4> output = storage_images[constants.image];
    output.Store(thread, float4(srgb_lin2encoded(value), 1));
}
