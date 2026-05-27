import shared.prefix_sum;
import shared.util;


struct Meshlet {
    float3 center;
    float radius;
    int8_t3 _cone_axis;
    int8_t _cone_cutoff;
    uint vertex_offset;
    uint triangle_offset;
    uint8_t vertex_count;
    uint8_t triangle_count;

    property float3 cone_axis {
        get {
            return float3(_cone_axis) / 127.0;
        }
    }

    property float cone_cutoff {
        get {
            return float(_cone_cutoff) / 127.0;
        }
    }
}

struct Instance {
    float3* positions;
    Meshlet* meshlets;
    uint8_t* triangles;
    uint* vertices;
    float4 rotation;
    float3 translation;
    float3 scale;


    uint image;
    uint flags;
    float radius;
    uint num_meshlets;

    float3 transform_point(float3 point) {
        return rotate_quat(rotation, point) * scale + translation;
    }
}

struct Uniforms {
    float4x4 camera;
    float4x4 view;
    float4 frustum;
    float3 camera_pos;
    float near_plane;
    uint2 extent;
    uint num_instances;
    uint vis;
}

struct PushConstants {
    Uniforms* uniforms;
    uint* prefix_sum_data;
    Instance* instances;
    uint swapchain;
    Atomic<uint>* visible_meshlets;
    uint2* meshlet_instances;
    uint3* dispatch;

    property PrefixSumBuffer prefix_sum {
        get {
            PrefixSumBuffer buffer;
            buffer.counter = (Atomic<uint64_t>*)prefix_sum_data;
            buffer.values = (PrefixSumValue*)&buffer.counter[1];
            return buffer;
        }
    }
}
