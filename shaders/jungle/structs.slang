import shared.prefix_sum;
import shared.util;

struct Instance {
    float3 translation;
    float scale;
    float3* positions;
    float2* uvs;
    uint* indices;
    uint num_indices;
    uint image;
    uint8_t flags;

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
    Atomic<uint64_t>* prefix_sum_counter;
    uint* prefix_sum_values;
    uint* offsets;
    DrawIndirectCommand* draw_command;
    Instance* instances;
    uint num_instances;

    property PrefixSumBuffer prefix_sum {
        get {
            PrefixSumBuffer buffer;
            buffer.counter = prefix_sum_counter;
            buffer.values = (PrefixSumValue*)prefix_sum_values;
            return buffer;
        }
    }
}
