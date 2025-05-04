import models;
import dispatches;
import prefix_sum;
import shared.noise;
import shared.util;

enum DebugMode {
    None,
    Triangles,
    Model,
    BaseColour,
    Normals,
    BaseNormals,
    MapNormals,
    Roughness,
    Metallic,
    RaytracedScene
};

struct UniformBuffer {
    float4x4 mat;
    float4x4 mat_inv;
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
    uint* half_size_shadow_buffer;
    int* _blue_noise_sobol;
    int* _blue_noise_scrambling_tile;
    int* _blue_noise_ranking_tile;
    uint64_t _acceleration_structure;
    uint num_instances;
    uint visbuffer;
    uint hdrbuffer;
    uint hdrbuffer_sampled;
    uint depthbuffer;
    uint prev_depthbuffer;
    uint swapchain_image;
    uint tonemap_lut_image;
    uint frame_index;
    DebugMode debug_mode;

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

    property uint2 half_extent {
        get {
            return div_ceil(extent, 2);
        }
    }

    property BlueNoiseData blue_noise {
        get {
            BlueNoiseData data;
            data.sobol = _blue_noise_sobol;
            data.scrambling_tile = _blue_noise_scrambling_tile;
            data.ranking_tile = _blue_noise_ranking_tile;
            return data;
        }
    }
};
