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
    var albedo = float3(0);

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

        let light_dir = SphericalDirectionalLightRayDirection(rng.rand2(), normalize(float3(2,10,2)), 0.1);

        if (instance.image != MAX_UINT_VALUE) {
            albedo = sample(instance.image, uv).rgb;
        } else {
            albedo = float3(1);
        }

        let geom_normal = normalize(cross(b - a, c - a));

        RayDesc desc;
        desc.Direction = light_dir;
        desc.Origin = offset_ray(position.val, geom_normal);
        desc.TMax = 10000.0;
        desc.TMin = 0.0;

        var query = RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH>();
        query.TraceRayInline(
            RaytracingAccelerationStructure(pc.tlas),
            RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
            0xff,
            desc
        );
        query.Proceed();

        radiance = max(dot(normal, light_dir), 0.0);

        if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            depth = query.CommittedRayT();
            radiance = 0.0;
        }
    }

    let roughness = 1.0;

    let normHitDist = REBLUR_FrontEnd_GetNormHitDist(depth, view_z, float3(3.0, 0.1, 20.0), roughness);
    let packed = REBLUR_FrontEnd_PackRadianceAndNormHitDist(radiance, normHitDist, false);

    WTexture2D<float4> radiance_tex = storage_images[pc.uniforms.radiance];
    radiance_tex.Store(thread, packed);

    WTexture2D<float3> albedo_tex = storage_images[pc.uniforms.albedo];
    albedo_tex.Store(thread, albedo);

    WTexture2D<float> view_z_tex = storage_images[pc.uniforms.linear_depth];
    view_z_tex.Store(thread, view_z);

    WTexture2D<float4> normal_roughness_tex = storage_images[pc.uniforms.normal_roughness];
    normal_roughness_tex.Store(thread, NRD_FrontEnd_PackNormalAndRoughness(normal, roughness, material_id));

    WTexture2D<float2> motion_tex = storage_images[pc.uniforms.motion];
    motion_tex.Store(thread, (thread_clip - prev_thread_clip)*-0.5);
}

[shader("compute")]
[NumThreads(8, 8, 1)]
void post_denoise(uint2 thread: SV_DispatchThreadID, uniform PushConstants pc) {
    if (any(thread >= pc.uniforms.extent)) {
        return;
    }

    Texture2D<float4> denoised = storage_images[pc.uniforms.denoised];

    Texture2D<float3> albedo = storage_images[pc.uniforms.albedo];

    let val = REBLUR_BackEnd_UnpackRadianceAndNormHitDist(denoised[thread]).xyz * albedo[thread];

    WTexture2D<float3> swapchain = storage_images[pc.swapchain];

    swapchain.Store(thread,srgb_lin2encoded(val));
}
