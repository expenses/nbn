import structs;
import shared.colour;
import fallback_bindings;
import shared.prefix_sum;

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

    let num_tris = pc.instances[thread].num_indices / 3;

    let value = prefix_sum_inclusive_append(pc.prefix_sum, thread, num_tris);

    pc.offsets[thread] = value.sum - num_tris;
}

[shader("vertex")]
Interp vertex(
    uint vertex_id : SV_VulkanVertexID,
    uint instance_id: SV_VulkanInstanceID,
    uniform PushConstants pc
) {
    let instance = pc.instances[instance_id];
    let index = instance.read_index(vertex_id);
    let position = instance.positions[index] * instance.scale + instance.translation;


    Interp interp;
    interp.position = mul(pc.camera, float4(position, 1.0));
    interp.value = vertex_id/3 + pc.offsets[instance_id];
    return interp;
}

[shader("pixel")]
float3 pixel(
    Interp interp,
    uniform PushConstants pc
) {
    let global_tri_id = interp.value;
    let val = prefix_sum_binary_search(pc.prefix_sum, global_tri_id);
    let instance_id = val.index;
    let tris_until_next = val.sum - interp.value;
    let instance = pc.instances[instance_id];
    let local_tri_id = instance.num_indices/3 - tris_until_next;

    let indices = uint3(
        instance.read_index(local_tri_id*3),
        instance.read_index(local_tri_id*3+1),
        instance.read_index(local_tri_id*3+2),
    );

    let pos = instance.positions[indices.x]  * instance.scale + instance.translation;// + instance.position.xyz,
    ;//model.positions[indices.y];// + instance.position.xyz,
    ;//model.positions[indices.z];//

    Sampler2D img = sampled_images[instance.image];

    return srgb_lin2encoded(pos);
}
