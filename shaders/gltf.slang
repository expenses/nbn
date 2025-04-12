module gltf;

public struct Material {
    public uint emissive_image;
    public uint base_colour_image;
    public uint metallic_roughness_image;
    public uint normal_image;
    public uint flags;
}

public struct BufferView {
    uint byte_offset;
}

public struct Accessor {
    uint buffer_view;
    uint count;
    uint byte_offset;
    uint flags;

    uint8_t *ptr(Gltf *gltf) {
        BufferView buffer_view = gltf.buffer_views[buffer_view];
        return gltf.buffer + buffer_view.byte_offset + byte_offset;
    };
}

public struct Primitive {
    uint indices;
    uint positions;
    uint uvs;
    uint normals;
    public uint material;

    public uint num_indices(Gltf *gltf) { return gltf.accessors[indices].count; }

    public uint read_index(Gltf *gltf, uint id) {
        let accessor = gltf.accessors[indices];

        let ptr = gltf.accessors[indices].ptr(gltf);

        if (accessor.flags == 1) {
            return uint(((uint32_t *)ptr)[id]);
        } else {
            return uint(((uint16_t *)ptr)[id]);
        }
    }

    public float3 *positions_ptr(Gltf *gltf) {
        return (float3 *)gltf.accessors[positions].ptr(gltf);
    }

    public float3 *normals_ptr(Gltf *gltf) {
        return (float3 *)gltf.accessors[normals].ptr(gltf);
    }

    public float2 *uvs_ptr(Gltf *gltf) {
        return (float2 *)gltf.accessors[uvs].ptr(gltf);
    }
}

public struct Gltf {
    public BufferView *buffer_views;
    public Accessor *accessors;
    public Primitive *primitives;
    public uint8_t *buffer;
    public Material *materials;
    public uint num_primitives;
}
