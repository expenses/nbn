import shared.prefix_sum;
import fallback_bindings;
import shared.colour;

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

struct Uniforms {
    float4x4 camera;
    float4x4 view;
    float4x4 inv_camera;
    float4x4 prev_camera;
};

struct PushConstants {
    uint2 extent;
    Uniforms* uniforms;
    uint image_idx;
    Ptr<Splat>* splats;
    uint64_t* bitmasks;
    Dispatch* dispatch;
    uint* point_to_splat;
    uint prev_color_idx;
    uint prev_depth_idx;
    uint color_idx;
    uint depth_idx;
    uint frame_index;
    uint num_splats;
    float tan_y;
    uint max_points;

    uint64_t* get_mask(uint2 coord) {
        return &bitmasks[coord.x + coord.y * extent.x];
    }

    float4 color_and_depth(uint2 coord) {
        let mask = *get_mask(coord);
    
        var color = srgb_encoded2lin(float3(178, 255, 255) / 255);
        var depth = 0.0;

        if (mask >> 32 != 0) {
            color = srgb_encoded2lin(float3(uint64_t3(mask, (mask >> 8), (mask >> 16)) & 255) / 255);
            depth = reinterpret<float>(uint(mask >> 32));
        }

        return float4(color, depth);
    }

    property WTexture2D<float3> image {
        get {
             return storage_images[image_idx];
        }
    }

    property WTexture2D<float3> color {
        get {
             return storage_images[color_idx];
        }
    }

    property WTexture2D<float> depth {
        get {
             return storage_images[depth_idx];
        }
    }

    property Sampler2D<float3> prev_color {
        get {
            return sampled_images[prev_color_idx];
        }
    }

    property Sampler2D<float> prev_depth {
        get {
            return sampled_images[prev_depth_idx];
        }
    }
}
