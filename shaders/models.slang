public struct Material {
    public uint emissive_image;
    public uint base_colour_image;
    public uint metallic_roughness_image;
    public uint normal_image;
    public uint flags;
}

public struct Meshlet {
    public uint vertex_offset;
    public uint triangle_offset;
    public uint vertex_count;
    public uint triangle_count;
};

[Flags]
enum ModelFlags {
    Indices32Bit
};

public struct Model {
    public float3* positions;
    public float2* uvs;
    public float3* normals;
    public uint* indices;

    Meshlet* meshlets;
    uint8_t* triangles;
    uint* vertices;

    public uint flags;
    public Material material;

    public uint read_index(uint vertex_index) {
        if ((flags & 1) == 1) {
            return indices[vertex_index];
        } else {
            uint16_t index = ((uint16_t*) indices)[vertex_index];
            return uint(index);
        }
    }

    public uint index(MeshletInstance meshlet_instance, uint vertex_index) {
        Meshlet meshlet = meshlet_instance.meshlet;
        uint micro_index = meshlet.vertex_offset + triangles[meshlet.triangle_offset + vertex_index];
        return vertices[micro_index];
    }
};

public struct Instance {
    public Model model;
    //uint model_index;
};

public struct MeshletInstance {
    public Meshlet meshlet;
    //public uint meshlet_index;
    public uint instance_index;
}
