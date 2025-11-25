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
    float3 position;
    float3 color;
    float intensity;
};

struct Uniforms {
    float4x4 view_inv;
    float4x4 proj_inv;
    float4x4 proj_view;
    uint64_t tlas;
    Model* model;
    float* light_values;
    Reservoir<int>* reservoirs;
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

    ThreadBlueNoise blue_noise(uint2 thread) {
        BlueNoiseData data;
        data.sobol = blue_noise_sobol;
        data.scrambling_tile = blue_noise_scrambling_tile;
        data.ranking_tile = blue_noise_ranking_tile;
        return ThreadBlueNoise(data, frame_index, thread);
    }
    
    uint thread_to_index(uint2 thread) {
        return thread.x + thread.y * extent.x;
    }

    Light get_light(uint index) {
        Light light;
        light.position[0] = light_values[index * 7 + 0];
        light.position[1] = light_values[index * 7 + 1];
        light.position[2] = light_values[index * 7 + 2];
        light.color[0] = light_values[index * 7 + 3];
        light.color[1] = light_values[index * 7 + 4];
        light.color[2] = light_values[index * 7 + 5];
        light.intensity = light_values[index * 7 + 6];
        return light;
    }
};

struct PushConstants {
    Uniforms* uniforms;
};
