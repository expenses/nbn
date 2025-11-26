import uniforms;
import shared.util;
import shared.constants;
import shared.barycentrics;
import shared.sampling_old;
import shared.noise;
import raytracing;
import fallback_bindings;

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

    RayDesc desc;
    desc.Origin = world_space_pos.xyz;
    desc.Direction = SphericalDirectionalLightRayDirection(
        uniforms.blue_noise.sample(thread, uniforms.frame_index),
        sun,
        0.01
    );
    desc.TMax = 1e6f;
    desc.TMin = 20.0;

    let in_shadow = trace_shadow_ray(uniforms, desc);

    let ballot = WaveActiveBallot(in_shadow);

    let row_width = div_ceil(uniforms.half_extent.x, 8);

    if (WaveIsFirstLane()) {
        uniforms.half_size_shadow_buffer
            [group_id.x + (group_id.y * 2) * row_width] = ballot.x;
        uniforms.half_size_shadow_buffer
            [group_id.x + (group_id.y * 2 + 1) * row_width] = ballot.y;
    }
}
