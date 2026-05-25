import structs;
import shared.colour;
import fallback_bindings;
import shared.prefix_sum;
import shared.util;
import shared.constants;
import shared.derivatives;
import shared.barycentrics;

struct Interp {
    float4 position: SV_Position;
    uint value;
}

[shader("compute")]
[NumThreads(1, 1, 1)]
void reset(uniform PushConstants pc) {
    *pc.prefix_sum.counter = 0;
}

[shader("compute")]
[NumThreads(64, 1, 1)]
void prefix_sum_instances(
    uint thread: SV_DispatchThreadID,
    uniform PushConstants pc
) {
    if ((thread >= pc.num_instances)) {
        return;
    }

    if (thread % 10 != 0) {
        return;
    }

    let num_indices = pc.instances[thread].num_indices;
    let num_tris = num_indices / 3;

    let value = prefix_sum_inclusive_append(pc.prefix_sum, thread, num_tris);

    DrawIndirectCommand* command = &pc.draw_commands[value.index];

    command.vertex_count = num_indices;
    command.instance_count = 1;
    command.first_vertex = value.sum - num_tris;
    command.first_instance = thread;
}

[shader("vertex")]
Interp vertex(
    uint vertex_id : SV_VertexID,
    uint offset : SV_StartVertexLocation,
    uint instance_id: SV_VulkanInstanceID,
    uniform PushConstants pc
) {
    let instance = pc.instances[instance_id];
    let index = instance.read_index(vertex_id);
    let position = instance.transform_point(instance.positions[index]);

    Interp interp;
    interp.position = mul(pc.camera, float4(position, 1.0));
    interp.value = vertex_id/3 + offset;
    return interp;
}

[shader("pixel")]
uint pixel(Interp interp) {
    return interp.value;
}

float3 resolve_inner(uint2 thread, PushConstants pc) {
    Texture2D<uint> visbuffer = storage_images[pc.vis];

    let global_tri_id = visbuffer[thread];

    if (global_tri_id == MAX_UINT_VALUE) {
        return float3(0.0);
    }

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
        pc.camera,
        instance.transform_point(instance.positions[indices.x]),
        instance.transform_point(instance.positions[indices.y]),
        instance.transform_point(instance.positions[indices.z]),
        thread,
        pc.extent,
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
        Sampler2D image = sampled_images[NonUniformResourceIndex(instance.image)];
        color = image.SampleGrad(uv.val, uv.dx, uv.dy).rgb;
    }

    return color;
}

[shader("compute")]
[NumThreads(8, 8, 1)]
void resolve(uint2 thread: SV_DispatchThreadID, uniform PushConstants pc) {
    if (any(thread >= pc.extent)) {
        return;
    }

    let value = resolve_inner(thread, pc);

    WTexture2D<float3> swapchain = storage_images[pc.swapchain];
    swapchain.Store(thread, srgb_lin2encoded(value));
}
