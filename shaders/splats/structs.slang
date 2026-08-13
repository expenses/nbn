import shared.prefix_sum;

struct Splat {
    float3 center;
    uint32_t color_opacity;
    uint32_t packed_scale;
    float4 rot;

    property float3 scale {
        get {
            let s = float3(packed_scale & 1023, (packed_scale >> 10) & 1023, packed_scale >> 20) / 1023.0;
            float log_min = -8;
            float log_max = 4;
            float3 log_s = log_min + s * (log_max - log_min);
            return exp(log_s);
        }
    }

    property float opacity {
        get {
            return float(color_opacity >> 24)/255.0;
        }
    }
}

struct Dispatch {
    uint3 dispatch_size;
    uint surviving_splats;
    uint num_points;
}

struct PerSplatData {
    float c0;
    float c1;
    float c2;
    float2 center_px;
    uint64_t packed;

    property float opacity {
        get {
            return float((packed >> 24)&255)/255.0;
        }
    }
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
    uint* point_to_splat;
}
