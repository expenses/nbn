import shared.noise;
import shared.restir;

[Flags]
enum ModelFlags {
    Indices32Bit
};

[Flags]
enum FrameFlags {
    TemporalReuse,
    OneRayPerPixel
};

struct Model {
    float3* positions;
    float2* uvs;
    float3* normals;
    uint* indices;
    uint* image_indices;
    ModelFlags flags;
    uint num_indices;

    uint read_index(uint vertex_index) {
        if (flags & ModelFlags::Indices32Bit) {
            return indices[vertex_index];
        } else {
            uint16_t index = ((uint16_t*)indices)[vertex_index];
            return uint(index);
        }
    }
};

struct Light {
    float3 emission;
    float3 position;
    float spotlight_angle_scale;
    float spotlight_angle_offset;
    float3 spotlight_direction;
};

struct Uniforms {
    float4x4 proj_view;
    float4x4 prev_proj_view;
    uint64_t tlas;
    Model* model;
    float[11]* lights;
    Reservoir<uint>* reservoirs;
    Reservoir<uint>* prev_reservoirs;
    int* blue_noise_sobol;
    int* blue_noise_ranking_tile;
    int* blue_noise_scrambling_tile;

    uint2 extent;
    uint hdr_image;
    uint swapchain_image;
    uint prims_image;
    uint num_lights;
    uint accum_index;
    uint frame_index;
    FrameFlags flags;

    float camera_pos_x;
    float camera_pos_y;
    float camera_pos_z;

    property float3 camera_pos {
        get {
            return float3(camera_pos_x, camera_pos_y, camera_pos_z);
        }
    }

    property BlueNoiseData blue_noise_data {
        get {
            BlueNoiseData data;
            data.sobol = blue_noise_sobol;
            data.scrambling_tile = blue_noise_scrambling_tile;
            data.ranking_tile = blue_noise_ranking_tile;
            return data;
        }
    }

    uint thread_to_index(uint2 thread) {
        return thread.x + thread.y * extent.x;
    }

    Light get_light(uint index) {
        return reinterpret<Light>(lights[index]);
    }

    float2 get_prev_position(float3 position) {
        let prev_pos = mul(prev_proj_view, float4(position, 1.0));
        return prev_pos.xy / prev_pos.w;
    }
};

struct PushConstants {
    Uniforms* uniforms;
};
