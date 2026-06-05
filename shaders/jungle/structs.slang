import shared.prefix_sum;
import shared.util;
import shared.structs;

struct Instance {
    float3* positions;
    float2* uvs;
    float3* normals;
    uint* indices;
    uint image;
    uint flags;

    uint read_index(uint vertex_id) {
        if (flags!=0) {
            return indices[vertex_id];
        } else {
            uint16_t index = ((uint16_t*)indices)[vertex_id];
            return uint(index);
        }
    }
}

struct Uniforms {
    float4x4 camera;
    float4x4 view;
    float4x4 view_inv;
    float4x4 proj_inv;
    float4x4 prev_camera;
    uint2 extent;
    uint vis;
    uint frame_index;
    uint linear_depth;
    uint normal_roughness;
    uint motion;
    uint radiance;
    uint denoised;
    uint albedo;
}

struct PushConstants {
    Uniforms* uniforms;
    Instance* instances;
    uint swapchain;
    uint64_t tlas;
}
