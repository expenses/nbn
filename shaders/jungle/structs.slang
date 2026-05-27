import shared.prefix_sum;
import shared.util;

struct Instance {
    float3* positions;
    float2* uvs;
    uint* indices;
    float4 rotation;
    float3 translation;
    float3 scale;
    uint num_indices;
    uint image;
    uint flags;
    uint padding;

    float3 transform_point(float3 point) {
        return rotate_quat(rotation, point) * scale + translation;
    }

    uint read_index(uint vertex_id) {
        if (flags!=0) {
            return indices[vertex_id];
        } else {
            uint16_t index = ((uint16_t*)indices)[vertex_id];
            return uint(index);
        }
    }
}

struct PushConstants {
    float4x4 camera;
    float3 camera_pos;
    uint* prefix_sum_data;
    DrawIndirectCommand* draw_commands;
    Instance* instances;
    uint* draw_counts;
    uint2 extent;
    uint num_instances;
    uint swapchain;
    uint vis;

    property PrefixSumBuffer prefix_sum {
        get {
            PrefixSumBuffer buffer;
            buffer.counter = (Atomic<uint64_t>*)prefix_sum_data;
            buffer.values = (PrefixSumValue*)&buffer.counter[1];
            return buffer;
        }
    }
}
