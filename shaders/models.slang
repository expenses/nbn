[Flags]
public enum MaterialFlags {
    AlphaClipped
};

public struct Material {
    public uint emissive_image;
    public uint base_colour_image;
    public uint metallic_roughness_image;
    public uint normal_image;
    public MaterialFlags flags;
}

public struct Meshlet {
    public float3 center;
    public float radius;
    public int8_t3 _cone_axis;
    public int8_t _cone_cutoff;
    public uint vertex_offset;
    public uint triangle_offset;
    public uint8_t vertex_count;
    public uint8_t triangle_count;

    public property float3 cone_axis {
        get {
            return float3(_cone_axis) / 127.0;
        }
    };

    public property float cone_cutoff {
        get {
            return float(_cone_cutoff) / 127.0;
        }
    };
};

[Flags]
public enum ModelFlags {
    Indices32Bit
};

public struct Model {
    public float3* positions;
    public float2* uvs;
    public float3* normals;
    public uint* indices;

    public Meshlet* meshlets;
    public uint8_t* triangles;
    public uint* vertices;

    public ModelFlags flags;
    public Material material;

    public uint num_meshlets;
    public uint num_indices;

    public uint read_index(uint vertex_index) {
        if (flags & ModelFlags::Indices32Bit) {
            return indices[vertex_index];
        } else {
            uint16_t index = ((uint16_t*) indices)[vertex_index];
            return uint(index);
        }
    }
};

public struct Instance {
    public uint4 _model_index_and_padding;
    public float4 position;

    public property uint model_index {
        get {
            return _model_index_and_padding.x;
        }
    }
};

public struct MeshletInstance {
    public Meshlet meshlet;
    public Instance instance;
}
