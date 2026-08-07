import shared.prefix_sum;

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

struct Dispatch {
    uint3 dispatch_size;
    uint surviving_splats;
    uint64_t prefix_sum_counter;
}

struct PerSplatData {
    float opacity;
    float c0;
    float c1;
    float c2;
    float2 center_px;
    uint64_t packed;
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
    Dispatch* dispatch;
    PerSplatData* splat_data;
    PrefixSumValue* prefix_sum_values;

    property PrefixSumBuffer prefix_sum {
        get {
            PrefixSumBuffer prefix_sum;
            prefix_sum.counter = (Atomic<uint64_t>*)&dispatch.prefix_sum_counter;
            prefix_sum.values = prefix_sum_values;
            return prefix_sum;
        }
    }
}
