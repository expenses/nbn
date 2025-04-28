import models;
import common;
import culling;
import uniforms;
import prefix_sum;

struct PushConstants {
    UniformBuffer* uniforms;
    uint is_alpha_clipped_meshlets;
};

struct Interp {
    float4 pos: SV_Position;
    // only for debugging
    float3 world_pos;
    float3 view_pos;
    float2 uv;
};

struct PerPrimitive {
    uint32_t visbuffer_value: SV_PrimitiveID;
};

bool coneCull(
    float3 center,
    float radius,
    float3 cone_axis,
    float cone_cutoff,
    float3 camera_position
) {
    return dot(normalize(center - camera_position), cone_axis)
        > cone_cutoff + radius / length(center - camera_position);
}

#define AS_GROUP_SIZE 64

struct Payload {
    uint MeshletIndices[AS_GROUP_SIZE];
};

groupshared Payload sPayload;

[shader("amplification")][numthreads(AS_GROUP_SIZE, 1, 1)] void task(
    uniform PushConstants push_constants,
    uint gtid : SV_GroupThreadID,
    uint meshlet_instance_index : SV_DispatchThreadID,
) {
    let uniforms = push_constants.uniforms;

    let is_alpha_clipped = push_constants.is_alpha_clipped_meshlets != 0;

    let num = is_alpha_clipped ? uniforms.num_alpha_clipped_meshlets()
                               : uniforms.num_opaque_meshlets();

    var visible = meshlet_instance_index < num;

    if (visible) {
        let prefix_sum_buffer =
            uniforms.get_prefix_sum_buffer(is_alpha_clipped);

        let val =
            prefix_sum_binary_search(prefix_sum_buffer, meshlet_instance_index);

        let instance = uniforms.instances[val.index];
        let model = uniforms.models[instance.model_index];

        uint32_t local_meshlet_index =
            meshlet_instance_index - (val.sum - model.num_meshlets);

        Meshlet meshlet = model.meshlets[local_meshlet_index];

        if (is_alpha_clipped) {
            meshlet_instance_index += uniforms.num_opaque_meshlets();
        }

        uniforms.meshlet_instances[meshlet_instance_index] = {
            meshlet,
            val.index
        };

        let view_center = mul(uniforms.view, float4(meshlet.center, 1.0)).xyz;

        visible &= !coneCull(
            meshlet.center,
            meshlet.radius,
            meshlet.cone_axis,
            meshlet.cone_cutoff,
            uniforms.camera_position
        );
        // near plane
        visible &= view_center.z - meshlet.radius < uniforms.near_plane;
        // left and right planes
        visible &= view_center.z * uniforms.frustum[1]
                - abs(view_center.x) * uniforms.frustum[0]
            > -meshlet.radius;
        // top and bottom planes
        visible &= view_center.z * uniforms.frustum[3]
                - abs(view_center.y) * uniforms.frustum[2]
            > -meshlet.radius;

        if (visible) {
            let index = WavePrefixSum(visible);
            sPayload.MeshletIndices[index] = meshlet_instance_index;
        }
    }

    let num_visible = WaveActiveCountBits(visible);

    DispatchMesh(num_visible, 1, 1, sPayload);
}

    [shader("mesh")][outputtopology("triangle")]
    [numthreads(128, 1, 1)] void vertex(
        uniform PushConstants push_constants,
        uint gtid : SV_GroupThreadID,
        uint gid : SV_GroupID,
        in payload Payload payload,
        out indices uint3 triangles[128],
        out vertices Interp vertices[64],
        out primitives PerPrimitive triangle_visbuffer_values[128],
    ) {
    let uniforms = push_constants.uniforms;
    var meshlet_instance_index = payload.MeshletIndices[gid];
    let meshlet_instance = uniforms.meshlet_instances[meshlet_instance_index];
    let meshlet = meshlet_instance.meshlet;

    let instance = meshlet_instance.instance;
    let model = uniforms.models[instance.model_index];

    SetMeshOutputCounts(meshlet.vertex_count, meshlet.triangle_count);

    if (gtid < meshlet.triangle_count) {
        triangles[gtid] = uint3(
            model.triangles[meshlet.triangle_offset + gtid * 3 + 0],
            model.triangles[meshlet.triangle_offset + gtid * 3 + 1],
            model.triangles[meshlet.triangle_offset + gtid * 3 + 2],
        );
        triangle_visbuffer_values[gtid].visbuffer_value =
            meshlet_instance_index << 7 | gtid;
    }

    if (gtid < meshlet.vertex_count) {
        uint index = model.vertices[meshlet.vertex_offset + gtid];
        let position = model.positions[index] + instance.position.xyz;
        vertices[gtid].pos = mul(uniforms.mat, float4(position, 1.0));
        vertices[gtid].world_pos = position;
        vertices[gtid].view_pos = mul(uniforms.view, float4(position, 1.0)).xyz;
        vertices[gtid].uv = model.uvs[index];
    }
}

[shader("pixel")] uint opaque_fragment(uint visbuffer_value : SV_PrimitiveID) {
    return visbuffer_value;
}

    [shader("pixel")] uint alpha_clipped_fragment(
        uniform PushConstants push_constants,
        Interp in,
        uint visbuffer_value : SV_PrimitiveID
    ) {
    let uniforms = push_constants.uniforms;

    let meshlet_instance_index = visbuffer_value >> 7;
    let meshlet_instance = uniforms.meshlet_instances[meshlet_instance_index];
    let instance = meshlet_instance.instance;
    let model = uniforms.models[instance.model_index];
    let material = model.material;

    if (material.base_colour_image != max_value) {
        float4 sample = DescriptorHandle<Sampler2D>(material.base_colour_image)
                            .Sample(in.uv);
        if (sample.a < 0.5) {
            discard;
        }
    }

    return visbuffer_value;
}
