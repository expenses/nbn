import shared.prefix_sum;
import fallback_bindings;

struct Splat {
    float3 center;
    uint32_t color_opacity;
    uint32_t packed_scale;
    uint32_t packed_rot;

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

    property float4 rot {
        get {
            let p = packed_rot;
            let qc  = (p >> 30) & 3;
            // inverse of quantize_10b: (bits - 511)/511, then undo the encode-time x sqrt(2)
            let sqrt_2 = sqrt(2);
            let v0 = float(int(p       & 1023) - 511) / 511.0 / sqrt_2;
            let v1 = float(int((p >> 10) & 1023) - 511) / 511.0 / sqrt_2;
            let v2 = float(int((p >> 20) & 1023) - 511) / 511.0 / sqrt_2;
            // largest component was dropped; recover it positive from unit length
            let large = sqrt(max(1.0 - (v0*v0 + v1*v1 + v2*v2), 0.0));

            var q = float4(0, 0, 0, 0);
            q[qc]            = large;
            q[(qc + 1) % 4]  = v0;
            q[(qc + 2) % 4]  = v1;
            q[(qc + 3) % 4]  = v2;
            // normalize in case of drift/degenerate inputs
            return normalize(q);
        }
    }
}

struct Dispatch {
    uint3 dispatch_size;
    uint num_points;
}

struct PushConstants {
    float4x4 camera;
    float4x4 view;
    uint2 extent;
    uint image_idx;
    Ptr<Splat>* splats;
    uint64_t* bitmasks;
    uint frame_index;
    uint num_splats;
    float tan_y;
    Dispatch* dispatch;
    uint* point_to_splat;
    uint max_points;

    uint64_t* get_mask(uint2 coord) {
        return &bitmasks[coord.x + coord.y * extent.x];
    }

    float4 srgb_color_and_depth(uint2 coord) {
        let mask = *get_mask(coord);

        if (mask >> 32 == 0) {
            return float4(178, 255, 255, 0) / 255;
        }

        return float4(
            float3(mask & 255, (mask >> 8) & 255, (mask >> 16) & 255) / 255,
            reinterpret<float>(mask >> 32)
        );
    }

    property WTexture2D<float3> image {
        get {
             return storage_images[image_idx];
        }
    }
}
