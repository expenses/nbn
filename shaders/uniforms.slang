import models;
import dispatches;

struct UniformBuffer {
    float4x4 mat;
    float4x4 view;
    float4x4 perspective;
    float near_plane;
    uint2 extent;
    Instance* instances;
    MeshletInstance* meshlet_instances;
    DispatchBuffer* dispatches;
    uint num_instances;
    uint visbuffer;
};