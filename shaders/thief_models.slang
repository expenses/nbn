
[Flags] public enum ModelFlags { Indices32Bit };

public struct Model {
  public
    float3* positions;
  public
    float2* uvs;
  public
    float3* normals;
  public
    uint* indices;

  public
    ModelFlags flags;
  public
    uint image;
  public
    uint num_indices;

  public uint read_index(uint vertex_index) {
        if (flags & ModelFlags::Indices32Bit) {
            return indices[vertex_index];
        } else {
            uint16_t index = ((uint16_t*)indices)[vertex_index];
            return uint(index);
        }
    }
};
