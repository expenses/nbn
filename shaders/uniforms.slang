import models;
import dispatches;
import prefix_sum;

struct UniformBuffer {
    float4x4 mat;
    float4x4 view;
    float4x4 perspective;
    float4 frustum;
    float3 camera_position;
    float near_plane;
    uint2 extent;
    Model* models;
    Instance* instances;
    MeshletInstance* meshlet_instances;
    DispatchBuffer* dispatches;
    uint64_t _acceleration_structure;
    uint num_instances;
    uint visbuffer;
    uint hdrbuffer;
    uint hdrbuffer_sampled;
    uint depthbuffer;
    uint swapchain_image;
    uint tonemap_lut_image;

    PrefixSumValue* opaque_prefix_sum_values;
    PrefixSumValue* alpha_clip_prefix_sum_values;

    PrefixSumBuffer get_prefix_sum_buffer(bool is_alpha_clipped) {
        return {
            &dispatches.counts[uint(is_alpha_clipped)],
            is_alpha_clipped ? alpha_clip_prefix_sum_values
                             : opaque_prefix_sum_values
        };
    }

    property RaytracingAccelerationStructure acceleration_structure {
        get {
            return RaytracingAccelerationStructure(_acceleration_structure);
        }
    }
};
