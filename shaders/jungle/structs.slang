import shared.prefix_sum;
import shared.util;
import shared.structs;
import shared.importance_sampling;
import fallback_bindings;

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
    uint view_z;
    uint normal_roughness;
    uint motion;
    uint radiance;
    uint denoised;
    uint albedo;
    uint swapchain;
    uint envmap;
    uint2 envmap_size;
}

struct PushConstants {
    Uniforms* uniforms;
    Instance* instances;
    Alias* alias_table;
    uint64_t tlas;

    property RWTexture2D<float4> radiance {
        get {
            return storage_images[uniforms.radiance];
        }
    }

    property RWTexture2D<float4> albedo {
        get {
            return storage_images[uniforms.albedo];
        }
    }

    property Texture2D<float4> denoised {
        get {
            return storage_images[uniforms.denoised];
        }
    }

    property WTexture2D<float3> swapchain {
        get {
             return storage_images[uniforms.swapchain];
        }
    }

    property WTexture2D<float3> view_z {
        get {
             return storage_images[uniforms.view_z];
        }
    }

    property WTexture2D<float4> normal_roughness {
        get {
             return storage_images[uniforms.normal_roughness];
        }
    }

    property WTexture2D<float2> motion {
        get {
             return storage_images[uniforms.motion];
        }
    }
}
