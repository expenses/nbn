
struct RenderComputePushConstants {
    float* output;
    uint* latent_textures;
    float* params;
    int resolution;
    uint channel_offset;
}

struct CalculateGradsPushConstants {
    Network* network;
    uint* latent_textures;
    uint* textures;
    int num_textures;
    uint* channel_bitmasks;
    uint iteration;
    uint batch_size;
    float* total;
    uint alpha_channel;
}

struct OptimizePushConstants {
    // For latent textures, this is actually half* but that can't be used due to missing StoragePushConstant16
    float* primal;
    float* grad;
    float* mean;
    float* variance;
    float learning_rate;
    uint64_t* bitmask;
    int num_values;
    float adam_m_factor;
    float adam_v_factor;
}

struct SumTexturesLossPushConstants {
    float* params;
    uint* latent_textures;
    uint* textures;
    uint* channel_bitmasks;
    int num_textures;
    int resolution;
    float* total;
    uint use_halfs;
    uint alpha_channel;
}

struct CompressBlocksPushConstants {
    Network* network;
    uint2* blocks;
    int texture_index;
}

struct CopyNetworkParamsPushConstants {
    Network* network;
    float* output;
    int write_halfs;
}

struct RenderToTexturePushConstants {
    uint* latent_textures;
    float* params;
    uint2 extent;
    uint texture_size;
    uint image;
    float scale;
    uint channel_offset;
    float exposure;
}

uint16_t quantize(float value, uint16_t bits) {
    let size = (1 << bits) - 1;
    return uint16_t(round(value * size));
}

uint16_t3 quantize(float3 value, uint16_t3 bits) {
    let size = (uint3(1) << bits) - 1;
    return uint16_t3(round(value * size));
}

[BackwardDerivative(as_quantized_f_bwd)]
float as_quantized(float value, uint16_t bits) {
    let size = (1 << bits) - 1;
    return float(quantize(value, bits)) / size;
}

// pass gradient straight through
void as_quantized_f_bwd(inout DifferentialPair<float> value, uint bits, float grad) {
    value = DifferentialPair<float>(value.p, value.d + grad);
}

[BackwardDerivative(as_quantized_v_bwd)]
float3 as_quantized(float3 value, uint16_t3 bits) {
    let size = (uint3(1) << bits) - 1;
    return float3(quantize(value, bits)) / size;
}

// pass gradient straight through
void as_quantized_v_bwd(inout DifferentialPair<float3> value, uint3 bits, float3 grad) {
    value = DifferentialPair<float3>(value.p, value.d + grad);
}

[Differentiable]
float sigmoid(float x) {
    let expX = exp(x);
    return expX / (expX + 1.0f);
}

[Differentiable]
float3 sigmoid(float3 x) {
    let expX = exp(x);
    return expX / (expX + 1.0f);
}

struct LatentTexture {
    // Actually half* but that can't be used due to missing StoragePushConstant16
    float* data;
    float* grads;
    uint64_t* bitmasks;
    uint num_blocks;

    half load_value(int index) {
        return ((half*) data)[index];
    }

    [BackwardDerivativeOf(load_value)]
    void load_value_bwd(int index, half grad) {
        InterlockedOr(bitmasks[index>>6], 1 << (index&63));
        InterlockedAdd(grads[index], grad);
    }

    [Differentiable]
    float load_alpha(int index) {
        return load_value(index+num_blocks*3*2);
    }

    [Differentiable]
    half3 load_endpoint(int index) {
        return half3(
            load_value(index*3),
            load_value(index*3+1),
            load_value(index*3+2)
        );
    }

    [Differentiable]
    float3 get_value(int mip, int2 coord) {
        let block_offset = get_mip_offset(mip) / 16;
        let block_size = get_mip_size(mip) / 4;
        let block_index = block_offset + (coord.x / 4) + (coord.y / 4) * block_size;
        let a = load_endpoint(block_index * 2);
        let b = load_endpoint(block_index * 2 + 1);
        let alpha = load_alpha(block_index * 16 + (coord.x % 4) + (coord.y % 4) * 4);
        return lerp(
            as_quantized(sigmoid(a), uint16_t3(5,6,5)),
            as_quantized(sigmoid(b), uint16_t3(5,6,5)),
            as_quantized(sigmoid(alpha), 2)
        );
    }

    int size;
    int num_mip_levels;

    int get_mip_size(int mip_level) {
        return size >> mip_level;
    }

    int get_mip_offset(int mip_level) {
        int offset = 0;
        for (int level = 0; level < mip_level; ++level) {
            offset += (size * size) >> level * 2;
        }
        return offset;
    }

    void mip_coords(float2 uv, int mip_size, out float2 fractional_uv, out int2 c0, out int2 c1) {
        let mip_uv = uv * float2(mip_size) - 0.5;
        c0 = ((int2(floor(mip_uv)) % mip_size) + mip_size) % mip_size;
        c1 = (c0 + 1) % mip_size;
        fractional_uv = mip_uv - floor(mip_uv);
    }

    [Differentiable]
    half3 sample(no_diff float2 uv, no_diff float mip_level) {
        mip_level = clamp(mip_level, 0.0, float(num_mip_levels - 1));
        int mip0 = int(mip_level);
        int mip1 = min(mip0 + 1, num_mip_levels - 1);
        float mip_fractional = mip_level - mip0;

        int mip0_size = get_mip_size(mip0);
        int mip1_size = get_mip_size(mip1);

        float2 mip0_fractional_uv;
        int2 mip0_c0;
        int2 mip0_c1;
        mip_coords(uv, mip0_size, mip0_fractional_uv, mip0_c0, mip0_c1);

        float2 mip1_fractional_uv;
        int2 mip1_c0;
        int2 mip1_c1;
        mip_coords(uv, mip1_size, mip1_fractional_uv, mip1_c0, mip1_c1);

        let v_00_m0 = get_value(mip0, int2(mip0_c0.x, mip0_c0.y));
        let v_10_m0 = get_value(mip0, int2(mip0_c1.x, mip0_c0.y));
        let v_01_m0 = get_value(mip0, int2(mip0_c0.x, mip0_c1.y));
        let v_11_m0 = get_value(mip0, int2(mip0_c1.x, mip0_c1.y));
        let v_00_m1 = get_value(mip1, int2(mip1_c0.x, mip1_c0.y));
        let v_10_m1 = get_value(mip1, int2(mip1_c1.x, mip1_c0.y));
        let v_01_m1 = get_value(mip1, int2(mip1_c0.x, mip1_c1.y));
        let v_11_m1 = get_value(mip1, int2(mip1_c1.x, mip1_c1.y));
        return half3(
            lerp(
                lerp(
                    lerp(v_00_m0, v_10_m0, mip0_fractional_uv.x),
                    lerp(v_01_m0, v_11_m0, mip0_fractional_uv.x),
                    mip0_fractional_uv.y
                ),
                lerp(
                    lerp(v_00_m1, v_10_m1, mip1_fractional_uv.x),
                    lerp(v_01_m1, v_11_m1, mip1_fractional_uv.x),
                    mip1_fractional_uv.y
                ),
                mip_fractional
            )
        );
    }
}

struct Network {
    LatentTexture textures[4];
    float* weights_and_biases;
    float* weights_and_biases_grad;
}
