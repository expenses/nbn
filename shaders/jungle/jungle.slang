import structs;
import shared.colour;
import fallback_bindings;
import shared.prefix_sum;
import shared.util;
import shared.constants;
import shared.derivatives;
import shared.barycentrics;
import shared.culling;

[shader("compute")]
[NumThreads(1, 1, 1)]
void reset(uniform PushConstants pc) {
    *pc.prefix_sum.counter = 0;
    *pc.dispatch = uint3(0,1,1);
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

    InterlockedMax(pc.dispatch[0][0], div_ceil(val.sum, 64));
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

groupshared uint[64] gs_global_meshlet_indices;

[shader("amplification")]
[numthreads(64, 1, 1)]
void task(
    uniform PushConstants pc,
    uint thread : SV_DispatchThreadID,
) {
    // Check that we're in range for the number of meshlets being drawn.

    var visible = thread < prefix_sum_total(pc.prefix_sum);

    if (visible) {
        // Binary search to find out what instance we're rasterizing meshlets for.

        let val = prefix_sum_binary_search(pc.prefix_sum, thread);
        let instance_index = val.index;
        let meshlet_count_prefix_sum_inc = val.sum;

        let instance = pc.instances[instance_index];
        let meshlet_count_prefix_sum_ex = meshlet_count_prefix_sum_inc - instance.num_meshlets;
        let local_meshlet_index = thread - meshlet_count_prefix_sum_ex;

        let meshlet = instance.meshlets[local_meshlet_index];

        let radius = meshlet.radius * max(instance.scale.x, max(instance.scale.y, instance.scale.z));
        let center = instance.transform_point(meshlet.center);

        // Attempt to cull the meshlet.

        visible &= coneCull(
            center,
            radius,
            rotate_quat(instance.rotation, meshlet.cone_axis),
            meshlet.cone_cutoff,
            pc.uniforms.camera_pos
        );
        let view_center = mul(pc.uniforms.view, float4(center, 1.0)).xyz;
        visible &= frustumCull(view_center, radius, pc.uniforms.near_plane, pc.uniforms.frustum);

        if (visible) {
            let index = WavePrefixSum(visible);
            gs_global_meshlet_indices[index] = thread;
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
