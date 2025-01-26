struct Material {
    uint emissive_image;
    uint base_colour_image;
    uint metallic_roughness_image;
}

struct BufferView {
    uint byte_offset;
}

struct Accessor {
    uint buffer_view;
    uint count;

    uint8_t* ptr(Gltf* gltf) {
        BufferView buffer_view = gltf.buffer_views[buffer_view];
        return gltf.buffer + buffer_view.byte_offset;
    };
}

struct Primitive {
    uint indices;
    uint positions;
    uint uvs;
    uint material;

    uint num_indices(Gltf* gltf) {
        return gltf.accessors[indices].count;
    }

    uint16_t* indices_ptr(Gltf* gltf) {
        return (uint16_t*)gltf.accessors[indices].ptr(gltf);
    }

    float3* positions_ptr(Gltf* gltf) {
        return (float3*)gltf.accessors[positions].ptr(gltf);
    }

    float2* uvs_ptr(Gltf* gltf) {
        return (float2*)gltf.accessors[uvs].ptr(gltf);
    }
}

struct Gltf {
    BufferView* buffer_views;
    Accessor* accessors;
    Primitive* primitives;
    uint8_t* buffer;
    Material* materials;
}
