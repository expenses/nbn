import shared.noise;
import shared.restir;
import shared.barycentrics;
import shared.importance_sampling;

[Flags] enum MaterialFlags {
    AlphaClipped
};

struct Material {
  uint emissive_image;
  uint base_colour_image;
  uint metallic_roughness_image;
  uint normal_image;
  MaterialFlags flags;
  float4 base_color_factor;
}

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
    uint* material_indices;
    ModelFlags flags;
    uint num_indices;
    Material* materials;

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
    uint64_t tlas_;
    uint64_t uv_tlas_;
    Model* model;
    Light* lights;
    float4* output;
    float4* temp;
    uint64_t* location_bitmasks;
    uint* triangle_indices;
    Alias* alias_table;

    property RaytracingAccelerationStructure tlas {
        get {
            return RaytracingAccelerationStructure(tlas_);
        }
    }

    property RaytracingAccelerationStructure uv_tlas {
        get {
            return RaytracingAccelerationStructure(uv_tlas_);
        }
    }

    uint2 extent;
    uint2 envmap_size;

    uint num_lights;

    uint sample_index;
    uint envmap;
};

struct ViewerConstants {
    float4x4 view_inv;
    float4x4 proj_inv;
    PushConstants* base;
    uint image;
    uint frame_index;
    uint lightmap;
}
