import models;
import common;

struct PushConstants {
    float4x4 mat;
    Instance* instances;
    MeshletInstance* meshlet_instances;
    uint instance_offset;
};

struct Interp {
    float4 pos : SV_Position;
    float2 uv;
};


struct PerPrimitive {
    uint32_t visbuffer_value : SV_PrimitiveID;
};

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(128, 1, 1)]
void vertex(
uniform PushConstants push_constants,
                 uint       gtid : SV_GroupThreadID,
                 uint       gid  : SV_GroupID,
    out indices  uint3      triangles[128],
    out vertices Interp vertices[64],
    out primitives PerPrimitive triangle_visbuffer_values[128],
) {
    let meshlet_instance_index = gid + push_constants.instance_offset;

    let meshlet_instance = push_constants.meshlet_instances[meshlet_instance_index];
    let meshlet = meshlet_instance.meshlet;
    let instance = push_constants.instances[meshlet_instance.instance_index];
    let model = instance.model;
    
    let cull = meshlet_instance_index > 500;
    
    if (cull) {
        SetMeshOutputCounts(0,0);
        return;
    }
    
    SetMeshOutputCounts(meshlet.vertex_count, meshlet.triangle_count);

    if (gtid < meshlet.triangle_count) {
        triangles[gtid] = uint3(
            model.triangles[meshlet.triangle_offset + gtid * 3 + 0],
            model.triangles[meshlet.triangle_offset + gtid * 3 + 1],
            model.triangles[meshlet.triangle_offset + gtid * 3 + 2],
        );
        triangle_visbuffer_values[gtid].visbuffer_value = meshlet_instance_index << 7 | gtid;
    }

    if (gtid < meshlet.vertex_count) {
        uint index = model.vertices[meshlet.vertex_offset + gtid];
        let position = model.positions[index];
        vertices[gtid].pos = mul(push_constants.mat, float4(position, 1.0));
        vertices[gtid].uv = model.uvs[index];
    }
}

[shader("pixel")]
uint opaque_fragment(uint visbuffer_value: SV_PrimitiveID) {
    return visbuffer_value;
}

[shader("pixel")]
uint alpha_clipped_fragment(
uniform PushConstants push_constants,

Interp in, uint visbuffer_value: SV_PrimitiveID) {
    let meshlet_instance_index = visbuffer_value >> 7;
    let meshlet_instance = push_constants.meshlet_instances[meshlet_instance_index];
    let instance = push_constants.instances[meshlet_instance.instance_index];
    let model = instance.model;
    let material = model.material;

    if (material.base_colour_image != max_value) {
        float4 sample = DescriptorHandle<Sampler2D>(material.base_colour_image).Sample(in.uv);
        if (sample.a < 0.5) {
            discard;
        }
    }

    return visbuffer_value;
}
