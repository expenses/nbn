
struct RenderComputePushConstants {
    float* output;
    Network* network;
    int resolution;
}

struct CalculateGradsPushConstants {
    Network* network;
    uint* textures;
    int num_textures;
    uint iteration;
}

struct OptimizerPushConstants {
    float* primal;
    float* grad;
    float* mean;
    float* variance;
    float learning_rate;
    int iteration;
    int num_values;
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
    float3* endpoints;
    float3* endpoints_grad;

    float* alpha;
    float* alpha_grad;

    float load_alpha(int index) {
        return alpha[index];
    }

    [BackwardDerivativeOf(load_alpha)]
    void load_alpha_bwd(int index, float grad) {
        InterlockedAdd(alpha_grad[index], grad);
    }

    float3 load_endpoint(int index) {
        return endpoints[index];
    }

    [BackwardDerivativeOf(load_endpoint)]
    void load_endpoint_bwd(int index, float3 grad) {
        InterlockedAdd(endpoints_grad[index][0], grad[0]);
        InterlockedAdd(endpoints_grad[index][1], grad[1]);
        InterlockedAdd(endpoints_grad[index][2], grad[2]);
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

    [Differentiable]
    half3 sample(no_diff float2 uv, no_diff float mip_level) {
        mip_level = clamp(mip_level, 0.0, float(num_mip_levels - 1));
        int mip0 = int(mip_level);
        int mip1 = min(mip0 + 1, num_mip_levels - 1);
        float mip_fractional = mip_level - mip0;

        int mip0_size = get_mip_size(mip0);
        int mip1_size = get_mip_size(mip1);

        let mip0_uv = uv * float2(mip0_size) - 0.5;
        let mip0_c0 = clamp(int2(mip0_uv), int2(0), mip0_size-1);
        let mip0_c1 = min(mip0_c0 + 1, mip0_size - 1);
        let mip0_fractional_uv = mip0_uv - mip0_c0;

        let mip1_uv = uv * float2(mip1_size) - 0.5;
        let mip1_c0 = clamp(int2(mip1_uv), int2(0), mip1_size-1);
        let mip1_c1 = min(mip1_c0 + 1, mip1_size - 1);
        let mip1_fractional_uv = mip1_uv - mip1_c0;

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
    LatentTexture latent_texture_1;
    LatentTexture latent_texture_2;
    LatentTexture latent_texture_3;
    LatentTexture latent_texture_4;
    float* weights_and_biases;
    float* weights_and_biases_grad;
}
