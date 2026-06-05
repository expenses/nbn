struct Splat {
    float3 dc;
    float3 center;
    float3 inv_scale_2;
    float4 rot;
    float3 sh1_0;
    float3 sh1_1;
    float3 sh1_2;
    float opacity_factor;
}

struct PushConstants {
    float4x4 view_inv;
    float4x4 proj_inv;
    uint64_t tlas;
    uint2 extent;
    uint image;
    Splat* splats;
    uint frame_index;
}
