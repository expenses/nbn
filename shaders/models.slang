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
    public uint vertex_offset;
    public uint triangle_offset;
    public uint vertex_count;
    public uint triangle_count;
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
    // todo: use index
    public Model model;
    public uint2 _padding;
    public float4 position;
    //uint model_index;
};

public struct MeshletInstance {
    // todo: embed instance.
    public Meshlet meshlet;
    //public uint meshlet_index;
    public uint instance_index;
}
