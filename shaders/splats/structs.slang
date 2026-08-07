struct Splat {
    float3 dc;
    float3 center;
    float3 scale;
    float4 rot;
    float opacity;
    //float3 sh1_0;
    //float3 sh1_1;
    //float3 sh1_2;
    //float opacity_factor;
}

struct PushConstants {
    float4x4 camera;
    float4x4 view;
    uint2 extent;
    uint image;
    Splat* splats;
    uint64_t* bitmasks;
    uint frame_index;
    uint num_splats;
    float tan_y;
}

struct PerSplatData {
    float opacity;
    float c0;
    float c1;
    float c2;
    float2 center_px;
    uint64_t packed;
}