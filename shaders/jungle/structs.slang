
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
    Instance* instances;
    uint instance_index;
}
