import structs;
import shared.colour;
import fallback_bindings;
import shared.prefix_sum;
import shared.util;
import shared.constants;
import shared.derivatives;
import shared.barycentrics;
import shared.culling;
import shared.sampling;
import shared.noise;
import shared.importance_sampling;
#include "NRD/Shaders/NRD.hlsli"

float4 sample(uint image_index, WithDerivatives<float2> uv) {
     Sampler2D image = sampled_images[NonUniformResourceIndex(image_index)];
     return image.SampleGrad(uv.val, uv.dx, uv.dy);
}

[shader("compute")]
[NumThreads(8, 8, 1)]
void raytrace(uint2 thread: SV_DispatchThreadID, uniform PushConstants pc) {
    if (any(thread >= pc.uniforms.extent)) {
        return;
    }

    var rng = SobolRng(thread, pc.uniforms.frame_index);

    let thread_clip = thread_to_clip(thread, pc.uniforms.extent);
    let origin = mul(pc.uniforms.view_inv, float4(0.0, 0.0, 0.0, 1.0)).xyz;
    let temp = mul(pc.uniforms.proj_inv, float4(thread_clip, 1.0, 1.0));
    let direction =
        mul(pc.uniforms.view_inv, float4(normalize(temp.xyz), 0.0)).xyz;

    RayDesc desc;
    desc.Direction = direction;
    desc.Origin = origin;
    desc.TMax = 10000.0;
    desc.TMin = 0.0;

    var query = RayQuery<RAY_FLAG_NONE>();
    query.TraceRayInline(
        RaytracingAccelerationStructure(pc.tlas),
        RAY_FLAG_NONE,
        0xff,
        desc
    );
    query.Proceed();

    var radiance = float3(0);
    var view_z = 1.0/0.0;
    var depth = 0.0;
    var normal = float3(0);
    uint material_id = 0;
    var albedo = float4(0);

    var prev_thread_clip = thread_clip;

    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        let instance_index = query.CommittedRayInstanceCustomIndex() + query.CommittedRayGeometryIndex();
        material_id = instance_index;
        let instance = pc.instances[instance_index];
        let prim = query.CommittedPrimitiveIndex();
        let mat = query.CommittedObjectToWorld3x4();
        let indices = uint3(
            instance.read_index(prim*3),
            instance.read_index(prim*3+1),
            instance.read_index(prim*3+2),
        );

        let a = instance.positions[indices.x];
        let b = instance.positions[indices.y];
        let c = instance.positions[indices.z];

        WithDerivatives<float3> position;
        let barycentrics = calculate_barycentrics_from_world_positions(
            pc.uniforms.camera,
            mul(mat, float4(a, 1)),
            mul(mat, float4(b, 1)),
            mul(mat, float4(c, 1)),
            thread,
            pc.uniforms.extent,
            position
        );

        let prev_clip = mul(pc.uniforms.prev_camera, float4(position.val, 1.0));
        prev_thread_clip = (prev_clip.xy / prev_clip.w);

        view_z = mul(pc.uniforms.view, float4(position.val, 1.0)).z;

        let uv = interpolate(
           barycentrics,
           instance.uvs[indices.x],
           instance.uvs[indices.y],
           instance.uvs[indices.z]
        );

        normal = normalize(interpolate(
           barycentrics,
           instance.normals[indices.x],
           instance.normals[indices.y],
           instance.normals[indices.z]
        ).val);

        if (instance.image != MAX_UINT_VALUE) {
            albedo = float4(sample(instance.image, uv).rgb, 1);
        } else {
            albedo = float4(1);
        }

        let envmap_sample = sample_envmap_by_rand(pc.alias_table, sampled_images[pc.uniforms.envmap], pc.uniforms.envmap_size, rng.rand2());

        let geom_normal = normalize(cross(b - a, c - a));

        RayDesc desc;
        desc.Direction = envmap_sample.direction;
        desc.Origin = offset_ray(position.val, geom_normal);
        desc.TMax = 10000.0;
        desc.TMin = 0.0;

        let n_dot_l = dot(normal, desc.Direction);

        if (n_dot_l > 0) {
            var query = RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH>();
            query.TraceRayInline(
                RaytracingAccelerationStructure(pc.tlas),
                RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
                0xff,
                desc
            );
            query.Proceed();


            if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                depth = query.CommittedRayT();
            } else {
                radiance = (envmap_sample.radiance / envmap_sample.pdf) * n_dot_l;
            }
        }
    } else {
        let prev_clip = mul(pc.uniforms.prev_camera, float4(direction, 0.0));
        prev_thread_clip = (prev_clip.xy / prev_clip.w);
        albedo = float4(sample_envmap_by_direction(pc.alias_table, sampled_images[pc.uniforms.envmap], pc.uniforms.envmap_size, direction).radiance, 0);
    }

    let roughness = 1.0;

    pc.radiance.Store(
        thread,
        REBLUR_FrontEnd_PackRadianceAndNormHitDist(
            radiance,
            REBLUR_FrontEnd_GetNormHitDist(depth, view_z, float3(3.0, 0.1, 20.0), roughness),
            false
        )
    );
    pc.albedo.Store(thread, albedo);
    pc.view_z[thread] = view_z;
    pc.normal_roughness.Store(thread, NRD_FrontEnd_PackNormalAndRoughness(normal, roughness, material_id));
    pc.motion.Store(thread, (thread_clip - prev_thread_clip)*-0.5);
}

[shader("compute")]
[NumThreads(8, 8, 1)]
void post_denoise(uint2 thread: SV_DispatchThreadID, uniform PushConstants pc) {
    if (any(thread >= pc.uniforms.extent)) {
        return;
    }

    let albedo = pc.albedo[thread];

    var val = albedo.rgb;
    if (albedo.a > 0.5) {
        val *= REBLUR_BackEnd_UnpackRadianceAndNormHitDist(pc.denoised[thread]).xyz;
    }

    let depth = pc.view_z[thread];

    let uv = (float2(thread)+0.5)/pc.uniforms.extent;
    let prev_uv = uv + pc.motion[thread];

    if (all(prev_uv >= 0) && all(prev_uv <= 1)) {
        let prev = pc.accum_read.SampleLevel(prev_uv, 0);
        val = lerp(val, prev.rgb, 0.9);
    }
    pc.accum_write.Store(thread, float4(val, depth));

    pc.swapchain.Store(thread,srgb_lin2encoded(val));
}
