import models;
import dispatches;

struct UniformBuffer {
    float4x4 mat;
    float4x4 view;
    float4x4 perspective;
    float4 frustum;
    float near_plane;
    uint2 extent;
    Model* models;
    Instance* instances;
    MeshletInstance* meshlet_instances;
    DispatchBuffer* dispatches;
    uint num_instances;
    uint visbuffer;
    float3 camera_position;
};
