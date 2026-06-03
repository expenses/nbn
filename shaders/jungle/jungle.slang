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

[shader("compute")]
[NumThreads(1, 1, 1)]
void reset(uniform PushConstants pc) {
    *pc.prefix_sum.counter = 0;
    *pc.dispatch = uint3(0,10,1);
    *pc.visible_meshlets = 0;
}

[shader("compute")]
[NumThreads(64, 1, 1)]
void prefix_sum_instances(
    uint thread: SV_DispatchThreadID,
    uniform PushConstants pc
) {
    if ((thread >= pc.uniforms.num_instances)) {
        return;
    }

    let instance = pc.instances[thread];

    let radius = instance.radius * max(instance.scale.x, max(instance.scale.y, instance.scale.z));

    let view_center = mul(pc.uniforms.view, float4(instance.translation, 1.0)).xyz;
    var visible = frustumCull(view_center, radius, pc.uniforms.near_plane, pc.uniforms.frustum);
    let dist = radius * 75;
    visible &= distance_squared(pc.uniforms.camera_pos, instance.translation) < (dist*dist);

    if (!visible) {
        return;
    }

    let val = prefix_sum_inclusive_append(pc.prefix_sum, thread, instance.num_meshlets);

    InterlockedMax(pc.dispatch[0][0], div_ceil(val.sum, 64*10));
}

float4 sample(uint image_index, WithDerivatives<float2> uv) {
     Sampler2D image = sampled_images[NonUniformResourceIndex(image_index)];
     return image.SampleGrad(uv.val, uv.dx, uv.dy);
}

float3 resolve_inner(uint2 thread, PushConstants pc) {
    Texture2D<uint> visbuffer = storage_images[pc.uniforms.vis];

    let global_tri_id = visbuffer[thread];

    if (global_tri_id == MAX_UINT_VALUE) {
        return float3(0.0);
    }

    /*
    let val = prefix_sum_binary_search(pc.prefix_sum, global_tri_id);
    let instance = pc.instances[val.index];
    let num_tris = instance.num_indices/3;
    let offset = val.sum - num_tris;
    let local_tri_id = global_tri_id - offset;

    let indices = uint3(
        instance.read_index(local_tri_id*3),
        instance.read_index(local_tri_id*3+1),
        instance.read_index(local_tri_id*3+2),
    );

    WithDerivatives<float3> position;
    let barycentrics = calculate_barycentrics_from_world_positions(
        pc.uniforms.camera,
        instance.transform_point(instance.positions[indices.x]),
        instance.transform_point(instance.positions[indices.y]),
        instance.transform_point(instance.positions[indices.z]),
        thread,
        pc.uniforms.extent,
        position,
    );

    let uv = interpolate(
        barycentrics,
        instance.uvs[indices.x],
        instance.uvs[indices.y],
        instance.uvs[indices.z]
    );

    //var normal = normalize(interpolate(
    //    bary,
    //    instance.normals[indices.x],
    //    instance.normals[indices.y],
    //    instance.normals[indices.z]
    //).val);

    float3 color = float3(1);

    if (instance.image != MAX_UINT_VALUE) {
        color = sample(instance.image, uv).rgb;
    }

    return color;
    */

    let global_meshlet_index = global_tri_id >> 8;
    Instance instance;
    let local_meshlet_index = pc.get_local_meshlet(global_meshlet_index, instance);
    let meshlet = instance.meshlets[local_meshlet_index];

    let tri = global_tri_id & 255;

    let indices = uint3(
        instance.vertices[meshlet.vertex_offset + instance.triangles[meshlet.triangle_offset + tri * 3 + 0]],
        instance.vertices[meshlet.vertex_offset + instance.triangles[meshlet.triangle_offset + tri * 3 + 1]],
        instance.vertices[meshlet.vertex_offset + instance.triangles[meshlet.triangle_offset + tri * 3 + 2]],
    );

    WithDerivatives<float3> position;
    let barycentrics = calculate_barycentrics_from_world_positions(
        pc.uniforms.camera,
        instance.transform_point(instance.positions[indices.x]),
        instance.transform_point(instance.positions[indices.y]),
        instance.transform_point(instance.positions[indices.z]),
        thread,
        pc.uniforms.extent,
        position,
    );

    let uv = interpolate(
        barycentrics,
        instance.uvs[indices.x],
        instance.uvs[indices.y],
        instance.uvs[indices.z]
    );

    //var normal = normalize(interpolate(
    //    bary,
    //    instance.normals[indices.x],
    //    instance.normals[indices.y],
    //    instance.normals[indices.z]
    //).val);

    float3 color = float3(1);

    if (instance.image != MAX_UINT_VALUE) {
        color = sample(instance.image, uv).rgb;
    }

    return color;
}

[shader("compute")]
[NumThreads(8, 8, 1)]
void resolve(uint2 thread: SV_DispatchThreadID, uniform PushConstants pc) {
    if (any(thread >= pc.uniforms.extent)) {
        return;
    }

    let value = resolve_inner(thread, pc);

    WTexture2D<float3> swapchain = storage_images[pc.swapchain];
    swapchain.Store(thread, srgb_lin2encoded(value));
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

    var color = float3(0);
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
            color = sample(instance.image, uv).rgb;
        } else {
            color = float3(1);
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

        albedo = color;

        let lighting = float(query.CommittedStatus() != COMMITTED_TRIANGLE_HIT);
        color = lighting;

        if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            depth = query.CommittedRayT();
        }
    }

    let roughness = 1.0;

    let normHitDist = REBLUR_FrontEnd_GetNormHitDist(depth, view_z, float3(3.0, 0.1, 20.0), roughness);
    let packed = REBLUR_FrontEnd_PackRadianceAndNormHitDist(color, normHitDist, false);

    WTexture2D<float4> radiance = storage_images[pc.uniforms.radiance];
    radiance.Store(thread, packed);

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

groupshared uint[64] gs_global_meshlet_indices;

[shader("amplification")]
[numthreads(64, 1, 1)]
void task(
    uniform PushConstants pc,
    uint2 thread : SV_DispatchThreadID,
) {
    // Check that we're in range for the number of meshlets being drawn.

    let global_meshlet_index = thread.y + thread.x * 10;

    var visible = global_meshlet_index < prefix_sum_total(pc.prefix_sum);

    if (visible) {
        // Find instance and meshlet via binary search.
        Instance instance;
        let local_meshlet_index = pc.get_local_meshlet(global_meshlet_index, instance);
        let meshlet = instance.meshlets[local_meshlet_index];

        // Attempt to cull the meshlet.

        let radius = meshlet.radius * max(instance.scale.x, max(instance.scale.y, instance.scale.z));
        let center = instance.transform_point(meshlet.center);

        visible &= coneCull(
            center,
            radius,
            rotate_quat(instance.rotation, meshlet.cone_axis),
            meshlet.cone_cutoff,
            pc.uniforms.camera_pos
        );
        let view_center = mul(pc.uniforms.view, float4(center, 1.0)).xyz;
        visible &= frustumCull(view_center, radius, pc.uniforms.near_plane, pc.uniforms.frustum);

        // Add to output if visible.
        if (visible) {
            let index = WavePrefixSum(visible);
            gs_global_meshlet_indices[index] = global_meshlet_index;
        }
    }

    let num_visible = WaveActiveCountBits(visible);
    DispatchMesh(num_visible, 1, 1, gs_global_meshlet_indices);
}

struct Interp {
    float4 pos: SV_Position;
}

struct PerPrimitive {
    uint32_t visbuffer_value: SV_PrimitiveID;
};

#define MESHLET_MAX_TRIS 256
#define MESHLET_MAX_VERTS 128
#define MESH_SHADER_GROUPSIZE 128

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(MESH_SHADER_GROUPSIZE, 1, 1)]
void mesh(
    uniform PushConstants pc,
    uint gtid : SV_GroupThreadID,
    uint gid : SV_GroupID,
    in payload uint[64] global_meshlet_indices,
    out indices uint3 triangles[MESHLET_MAX_TRIS],
    out vertices Interp vertices[MESHLET_MAX_VERTS],
    out primitives PerPrimitive triangle_visbuffer_values[MESHLET_MAX_TRIS],
) {
    let global_meshlet_index = global_meshlet_indices[gid];
    Instance instance;
    let local_meshlet_index = pc.get_local_meshlet(global_meshlet_index, instance);
    let meshlet = instance.meshlets[local_meshlet_index];

    SetMeshOutputCounts(meshlet.vertex_count, meshlet.triangle_count);

    // Using min clamping seems worth it according to
    // https://interplayoflight.wordpress.com/2025/05/05/meshlets-and-mesh-shaders/
    // todo: profile?

    [ForceUnroll]
    for (int i = 0; i < MESHLET_MAX_TRIS; i+= MESH_SHADER_GROUPSIZE) {
        let tri = min(i + gtid, meshlet.triangle_count - 1);

        triangles[tri] = uint3(
            instance.triangles[meshlet.triangle_offset + tri * 3 + 0],
            instance.triangles[meshlet.triangle_offset + tri * 3 + 1],
            instance.triangles[meshlet.triangle_offset + tri * 3 + 2],
        );
        // note: if we run out of memory here, could just add a R8 buffer
        // and store tris in that.
        triangle_visbuffer_values[tri].visbuffer_value =
            global_meshlet_index << 8 | tri;
    }

    [ForceUnroll]
    for (int i = 0; i < MESHLET_MAX_VERTS; i += MESH_SHADER_GROUPSIZE) {
        let vert = min(i + gtid, meshlet.vertex_count - 1);
        uint index = instance.vertices[meshlet.vertex_offset + vert];
        let position = instance.transform_point(instance.positions[index]);
        vertices[vert].pos = mul(pc.uniforms.camera, float4(position, 1.0));
    }
}

[shader("pixel")]
uint pixel(uint visbuffer_value : SV_PrimitiveID) {
    return visbuffer_value;
}
