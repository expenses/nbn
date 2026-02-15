import shared.noise;
import shared.restir;
import shared.barycentrics;

[Flags]
enum ModelFlags {
    Indices32Bit
};

struct Model {
    float3* positions;
    float2* uvs;
    float2* uv2s;
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

    uint3 read_indices(uint prim) {
        return uint3(
            read_index(prim * 3 + 0),
            read_index(prim * 3 + 1),
            read_index(prim * 3 + 2),
        );
    }
};

struct Light {
    float3 emission;
    float3 position;
    float spotlight_angle_scale;
    float spotlight_angle_offset;
    float3 spotlight_direction;
};

struct PushConstants {
    uint64_t tlas;
    Model* model;
    float[11]* lights;
    int* blue_noise_sobol;
    int* blue_noise_ranking_tile;
    int* blue_noise_scrambling_tile;

    uint2 extent;

    uint output;
    uint num_lights;

    uint positions;
    uint normals;
    uint temp;

    uint sample_index;
    uint samples_per_iter;
    uint total_samples;

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
};
