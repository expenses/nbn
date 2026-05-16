import slang.neural;
import ntc.structs;
import ntc.adam;
import shared.noise;
import fallback_bindings;
import shared.util;
import shared.colour;

static const int SubgroupSize = 64;
static const int SubgroupCount = 1;

typealias ShMemSize = SharedMemorySize<float, TargetEnum.SPIR_V, ExecutionMode.Training, SubgroupSize, SubgroupCount>;
typealias ShMemSizeForNetwork = ShMemSize.OfLayer3<16, 64, 64, 16>;

typealias Address = PointerAddress<float>;
typealias InputVec = WaveTangledVector<float, ShMemSizeForNetwork, 16, SubgroupSize>;
typealias HiddenVec = WaveTangledVector<float, ShMemSizeForNetwork, 64, SubgroupSize>;
typealias OutputVec = WaveTangledVector<float, ShMemSizeForNetwork, 16, SubgroupSize>;

typealias Layer0 = FFLayer<float, InputVec, HiddenVec, LinearLayout, ReLU<float>>;
typealias Layer1 = FFLayer<float, HiddenVec, HiddenVec, LinearLayout, ReLU<float>>;
typealias Layer2 = FFLayer<float, HiddenVec, OutputVec, LinearLayout, IdentityActivation<float>>;

float[16] vec_to_floats_inner(OutputVec vec) {
    float[16] output;
    for (int i = 0; i < 16; ++i) {
        output[i] = vec[i];
    }
    return output;
}

[Differentiable]
float[16] vec_to_floats(OutputVec v) {
    return no_diff vec_to_floats_inner(v);
}

[BackwardDerivativeOf(vec_to_floats)]
void vec_to_floats_bwd(inout DifferentialPair<OutputVec> pair, float[16] grads) {
    var pair_grads = pair.d;
    for (int i = 0; i < 16; ++i) {
        pair_grads[i] += grads[i];
    }
    pair = diffPair(pair.p, pair_grads);
}

[Differentiable]
half[16] eval_from_inputs(half[16] inputs, Address address) {
    InputVec input;
    for (int i = 0; i < 16; ++i) {
        input[i] = inputs[i];
    }

    let h0 = Layer0(ReLU<float>()).eval<Address>(input, address);
    address = address.getOffset(Layer0.ParameterCount);
    let h1 = Layer1(ReLU<float>()).eval<Address>(h0, address);
    address = address.getOffset(Layer1.ParameterCount);
    let output = Layer2(IdentityActivation<float>()).eval<Address>(h1, address);

    let floats = vec_to_floats(output);
    half[16] outputs;
    for (int i = 0; i < 16; i++) {
        outputs[i] = half(floats[i]);
    }
    return outputs;
}

half[16] eval_half_from_inputs(half[16] inputs, half* params) {
    var address = PointerAddress<half>(params);

    typealias ShMemSize = SharedMemorySize<half, TargetEnum.SPIR_V, ExecutionMode.Inference, SubgroupSize, SubgroupCount>;
    typealias ShMemSizeForNetwork = ShMemSize.OfLayer3<16, 64, 64, 16>;

    typealias InputVec = WaveTangledVector<half, ShMemSizeForNetwork, 16, SubgroupSize>;
    typealias HiddenVec = WaveTangledVector<half, ShMemSizeForNetwork, 64, SubgroupSize>;
    typealias OutputVec = WaveTangledVector<half, ShMemSizeForNetwork, 16, SubgroupSize>;

    typealias Layer0 = FFLayer<half, InputVec, HiddenVec, LinearLayout, ReLU<half>>;
    typealias Layer1 = FFLayer<half, HiddenVec, HiddenVec, LinearLayout, ReLU<half>>;
    typealias Layer2 = FFLayer<half, HiddenVec, OutputVec, LinearLayout, IdentityActivation<half>>;

    InputVec input;
    for (int i = 0; i < 16; ++i) {
        input[i] = inputs[i];
    }

    let h0 = Layer0(ReLU<half>()).eval<PointerAddress<half>>(input, address);
    address = address.getOffset(Layer0.ParameterCount);
    let h1 = Layer1(ReLU<half>()).eval<PointerAddress<half>>(h0, address);
    address = address.getOffset(Layer1.ParameterCount);
    let output = Layer2(IdentityActivation<half>()).eval<PointerAddress<half>>(h1, address);

    half[16] outputs;
    for (int i = 0; i < 16; i++) {
        outputs[i] = half(output[i]);
    }
    return outputs;
}

half[16] inputs_from_textures(
    uint* textures,
    float2 uv,
    float4 uv_grad,
    int size
) {
    Sampler2D t_1 = sampled_images[NonUniformResourceIndex(textures[0])];
    Sampler2D t_2 = sampled_images[NonUniformResourceIndex(textures[1])];
    Sampler2D t_3 = sampled_images[NonUniformResourceIndex(textures[2])];
    Sampler2D t_4 = sampled_images[NonUniformResourceIndex(textures[3])];

    let s_1 = half4(t_1.SampleGrad(uv, uv_grad.xy, uv_grad.zw));
    let s_2 = half4(t_2.SampleGrad(uv+(0.5/size), uv_grad.xy, uv_grad.zw));
    let s_3 = half4(t_3.SampleGrad(uv, uv_grad.xy, uv_grad.zw));
    let s_4 = half4(t_4.SampleGrad(uv+(0.5/(size/2)), uv_grad.xy, uv_grad.zw));

    half inputs[16] = {
        s_1[0],
        s_1[1],
        s_1[2],
        s_2[0],
        s_2[1],
        s_2[2],
        s_3[0],
        s_3[1],
        s_3[2],
        s_4[0],
        s_4[1],
        s_4[2],
        0.0h,
        0.0h,
        0.0h,
        0.0h
    };
    return inputs;
}


half[16] inputs_from_textures_mip(
    uint* textures,
    float2 uv,
    float mip,
    int size
) {
    Sampler2D t_1 = sampled_images[NonUniformResourceIndex(textures[0])];
    Sampler2D t_2 = sampled_images[NonUniformResourceIndex(textures[1])];
    Sampler2D t_3 = sampled_images[NonUniformResourceIndex(textures[2])];
    Sampler2D t_4 = sampled_images[NonUniformResourceIndex(textures[3])];

    let s_1 = half4(t_1.SampleLevel(uv, mip));
    let s_2 = half4(t_2.SampleLevel(uv+(0.5/size), mip));
    let s_3 = half4(t_3.SampleLevel(uv, mip-1));
    let s_4 = half4(t_4.SampleLevel(uv+(0.5/(size/2)), mip-1));

    half inputs[16] = {
        s_1[0],
        s_1[1],
        s_1[2],
        s_2[0],
        s_2[1],
        s_2[2],
        s_3[0],
        s_3[1],
        s_3[2],
        s_4[0],
        s_4[1],
        s_4[2],
        0.0h,
        0.0h,
        0.0h,
        0.0h
    };
    return inputs;
}


[Differentiable]
half[16] inputs_from_network(Network network, no_diff float2 uv, no_diff float mip_level) {
    let s_1 = network.textures[0].sample(uv, mip_level);
    let s_2 = network.textures[1].sample(uv+(0.5/network.textures[1].size), mip_level);
    let s_3 = network.textures[2].sample(uv, mip_level-1);
    let s_4 = network.textures[3].sample(uv+(0.5/network.textures[3].size), mip_level-1);
    half inputs[16] = {
        s_1[0],
        s_1[1],
        s_1[2],
        s_2[0],
        s_2[1],
        s_2[2],
        s_3[0],
        s_3[1],
        s_3[2],
        s_4[0],
        s_4[1],
        s_4[2],
        0.0h,
        0.0h,
        0.0h,
        0.0h
    };
    return inputs;
}

[Differentiable]
half relu(half x) {
    return max(x, half(0.0));
}

float[16] sample_references(
    float2 uv,
    float mip_level,
    uint* textures,
    int num_images,
    uint8_t* channel_bitmasks,
    inout uint num_channels
) {
    float[16] values;
    for (int i = 0; i < 16; i++) {
        values[i] = 0.0;
    }
    var offset = 0;
    for (int i = 0; i < num_images; i++) {
        Sampler2D<float4> texture = sampled_images[NonUniformResourceIndex(textures[i])];
        float4 sample = texture.SampleLevel(uv, mip_level);

        for (int j = 0; j < 4; ++j) {
            if (!bool(channel_bitmasks[i] & (1 << j))) {
                continue;
            }

            values[num_channels] = sample[j];
            num_channels ++;
        }
    }
    return values;
}

[Differentiable]
float calc_loss(float value, float reference, const bool l2) {
    let diff = value - reference;

    if (l2) {
        return diff * diff;
    } else {
        return abs(diff);
    }
}

[Differentiable]
float[16] loss(
    half[16] evaluated,
    no_diff float2 uv,
    no_diff float mip_level,
    no_diff uint* textures,
    no_diff int num_images,
    no_diff uint8_t* channel_bitmasks,
    no_diff float* total,
    no_diff bool l2,
) {
    uint num_channels = 0;
    let reference_values = (no_diff sample_references(
        uv,
        mip_level,
        textures,
        num_images,
        channel_bitmasks,
        num_channels
    ));

    let alpha_channel = 3;
    let is_alpha = reference_values[alpha_channel] == 0.0;

    float[16] error;
    [ForceUnroll]
    for (int i = 0; i < 16; ++i) {
        error[i] = calc_loss(evaluated[i], reference_values[i], l2);
        // Let the network focus on channels we care about.
        if (i >= num_channels || (is_alpha && i != alpha_channel)) {
            error[i] = 0.0;
        }
    }
    if (total) {
        (no_diff increment_loss(total, error));
    }
    return error;
}

void increment_loss(float* total, float[16] loss) {
    float local_sum = 0.0;
    for (int i = 0; i < 16; ++i) {
        local_sum += loss[i];
    }
    let wave_sum = WaveActiveSum(local_sum);
    if (WaveIsFirstLane()) {
        InterlockedAdd<float>(*total, wave_sum);
    }
}

[Differentiable]
float[16] loss_from_coord(
    no_diff float2 uv,
    no_diff float mip_level,
    no_diff uint* textures,
    Network network,
    Address address,
    no_diff int num_images,
    no_diff uint8_t* channel_bitmasks,
    no_diff float* total,
    no_diff bool l2
) {
    return loss(
        eval_from_inputs(inputs_from_network(network, uv, mip_level), address),
        uv,
        mip_level,
        textures,
        num_images,
        channel_bitmasks,
        total,
        l2
    );
}

[Differentiable]
float[16] loss_from_inputs(
    no_diff float2 uv,
    no_diff float mip_level,
    no_diff uint* textures,
    no_diff half[16] inputs,
    Address address,
    no_diff int num_images,
    no_diff uint8_t* channel_bitmasks,
    no_diff float* total,
    no_diff bool l2
) {
    return loss(
        eval_from_inputs(inputs, address),
        uv,
        mip_level,
        textures,
        num_images,
        channel_bitmasks,
        total,
        l2
    );
}

[shader("compute")]
[NumThreads(64, 1, 1)]
void optimizer_step(
    uint thread: SV_DispatchThreadID,
    uniform OptimizePushConstants pc
) {
    if ((thread >= pc.num_values)) {
        return;
    }

    pc.primal[thread] += adam_step(
        pc.grad[thread],
        pc.mean[thread],
        pc.variance[thread],
        pc.learning_rate,
        pc.adam_m_factor,
        pc.adam_v_factor,
    );
    // Clear gradients
    pc.grad[thread] = 0.0;
}

[shader("compute")]
[NumThreads(64, 1, 1)]
void optimize_latent_textures_dense(
    uint thread: SV_DispatchThreadID,
    uniform OptimizePushConstants pc,
) {
    if ((thread >= pc.num_values)) {
        return;
    }

    uint64_t* mask = &pc.bitmask[thread>>6];

    if (!bool(*mask & (uint64_t(1) << (thread&63)))) {
        return;
    }

    if (pc.grad[thread] == 0.0) {
        return;
    }

    // Take an adam step
    ((half*) pc.primal)[thread] += half(adam_step(
        pc.grad[thread],
        pc.mean[thread],
        pc.variance[thread],
        pc.learning_rate,
        pc.adam_m_factor,
        pc.adam_v_factor,
    ));
    // Clear gradients
    pc.grad[thread] = 0.0;
    *mask = 0;
}

[shader("compute")]
[NumThreads(64, 1, 1)]
void optimize_latent_textures(
    uint thread: SV_DispatchThreadID,
    uniform OptimizePushConstants pc,
) {
    if ((thread >= div_ceil(pc.num_values, 64))) {
        return;
    }

    var bitmask = pc.bitmask[thread];

    while (bool(bitmask)) {
        let bit = firstbitlow(bitmask);
        bitmask &= ~(uint64_t(1) << bit);
        let i = thread*64 + bit;

        if (pc.grad[i] == 0.0) {
            continue;
        }

        // Take an adam step
        ((half*) pc.primal)[i] += half(adam_step(
            pc.grad[i],
            pc.mean[i],
            pc.variance[i],
            pc.learning_rate,
            pc.adam_m_factor,
            pc.adam_v_factor,
        ));
        // Clear gradients
        pc.grad[i] = 0.0;
    }

    pc.bitmask[thread] = 0;
}

[shader("compute")]
[NumThreads(64, 1, 1)]
void sum_textures_loss(
    uint thread: SV_DispatchThreadID,
    uniform SumTexturesLossPushConstants pc
) {
    if (thread >= (pc.resolution*pc.resolution)) {
        return;
    }

    let coord = uint2(thread % pc.resolution, thread/pc.resolution);

    let uv = (float2(coord) + 0.5f) / float2(pc.resolution);
    let mip_level = 0.0;

    let inputs = inputs_from_textures(pc.latent_textures, uv, mip_level, pc.resolution);

    let evaluated = bool(pc.use_halfs)
        ? eval_half_from_inputs(inputs, (half*)pc.params)
        : eval_from_inputs(inputs, PointerAddress<float>(pc.params));

    loss(
        evaluated,
        uv,
        mip_level,
        pc.textures,
        pc.num_textures,
        (uint8_t*)pc.channel_bitmasks,
        pc.total,
        true
    );
}

float3 get_uv_and_mip(
    uint thread,
    CalculateGradsPushConstants pc
) {
    let coord = uint2(thread % pc.batch_size, thread/pc.batch_size);
    var rng = SobolRng(coord, pc.iteration);
    var uv = float3((coord + rng.rand2()) / float2(pc.batch_size), 0);
    let num_mip_levels = pc.network.textures[0].num_mip_levels;
    if (rng.rand() < 0.1f) {
        uv.z = (rng.rand() * (num_mip_levels+1))-1; // uniform
    } else {
        float u = max(rng.rand(), 1e-6f);
        uv.z = clamp(-log(u) / log(4.0f), 0.0f, float(num_mip_levels - 1));
    }
    return uv;
}

[shader("compute")]
[NumThreads(64, 1, 1)]
void calculate_grads(
    uint thread: SV_DispatchThreadID,
    uniform CalculateGradsPushConstants pc
) {
    if (thread >= (pc.batch_size * pc.batch_size)) {
        return;
    }

    let uv_mip = get_uv_and_mip(thread, pc);

    let address = DifferentialPtrPair<Address>(
        Address(pc.network.weights_and_biases),
        Address(pc.network.weights_and_biases_grad)
    );

    // Backpropagate through the loss
    float[16] gradient_seed;
    for (int i = 0; i < 16; ++i) {
        gradient_seed[i] = 1.0f;
    }
    bwd_diff(loss_from_coord)(
        uv_mip.xy,
        uv_mip.z,
        pc.textures,
        *pc.network,
        address,
        pc.num_textures,
        (uint8_t*) pc.channel_bitmasks,
        pc.total,
        false,
        gradient_seed
    );
}

[shader("compute")]
[NumThreads(64, 1, 1)]
void calculate_mlp_grads_only(
    uint thread: SV_DispatchThreadID,
    uniform CalculateGradsPushConstants pc
) {
    if (thread >= (pc.batch_size*pc.batch_size)) {
        return;
    }

    let uv_mip = get_uv_and_mip(thread, pc);

    let address = DifferentialPtrPair<Address>(
        Address(pc.network.weights_and_biases),
        Address(pc.network.weights_and_biases_grad)
    );

    // Backpropagate through the loss
    float[16] gradient_seed;
    for (int i = 0; i < 16; ++i) {
        gradient_seed[i] = 1.0f;
    }

    let inputs = inputs_from_textures_mip(pc.latent_textures, uv_mip.xy, uv_mip.z, pc.network.textures[0].size);

    bwd_diff(loss_from_inputs)(
        uv_mip.xy,
        uv_mip.z,
        pc.textures,
        inputs,
        address,
        pc.num_textures,
        (uint8_t*) pc.channel_bitmasks,
        pc.total,
        false,
        gradient_seed
    );
}

[shader("compute")]
[NumThreads(64, 1, 1)]
void compress_blocks(
    uint thread: SV_DispatchThreadID,
    uniform CompressBlocksPushConstants pc
) {
    let texture = pc.network.textures[pc.texture_index];

    if (thread >= texture.num_blocks) {
        return;
    }

    let blocks = (uint16_t4*) pc.blocks;

    blocks[thread] = compress_latent_texture(texture, thread);
}

[shader("compute")]
[NumThreads(64, 1, 1)]
void copy_network_params(
    uint thread: SV_DispatchThreadID,
    uniform CopyNetworkParamsPushConstants pc
) {
    let num_values = 6288;

    if (thread >= num_values) {
        return;
    }

    let param = pc.network.weights_and_biases[thread];

    if (bool(pc.write_halfs)) {
        let output = (half*) pc.output;
        output[thread] = half(param);
    } else {
        pc.output[thread] = param;
    }
}

uint16_t4 compress_latent_texture(LatentTexture texture, int block) {
    let endpoint_a = quantize(sigmoid(texture.load_endpoint(block * 2)), uint16_t3(5,6,5));
    let endpoint_b = quantize(sigmoid(texture.load_endpoint(block * 2 + 1)), uint16_t3(5,6,5));

    let e1 = endpoint_a.b | (endpoint_a.g << 5) | (endpoint_a.r << 11);
    let e2 = endpoint_b.b | (endpoint_b.g << 5) | (endpoint_b.r << 11);
    // bc1 has a seperate mode if e2 >= e1 that we want to avoid
    let max_e = max(e1, e2);
    let min_e = min(e1, e2);

    uint alphas = 0;
    for (int i = 0; i < 16; ++i) {
        var alpha = uint(quantize(sigmoid(texture.load_alpha(block * 16 + i)), 2));

        // If the endpoints were switched, make sure we flip the bits
        if (max_e != e1) {
            alpha = 3 - alpha;
        // If the end points are identical then we're in mode 2 and should just use the first endpoint
        } else if (max_e == min_e) {
            alpha = 0;
        }

        // bc1 has the second bit as the bit that uses endpoint 2.
        if (alpha == 3) {
            alpha = 1;
        } else if (alpha == 1 || alpha == 2) {
            alpha ++;
        }
        alphas |= alpha << i * 2;
    }

    return uint16_t4(
        max_e,
        min_e,
        uint16_t(alphas),
        uint16_t(alphas >> 16),
    );
}

[shader("compute")]
[NumThreads(64, 1, 1)]
void render_compute(
    uint thread : SV_DispatchThreadID,
    uniform RenderComputePushConstants pc
) {
    if (any(thread >= pc.resolution*pc.resolution)) {
        return;
    }

    let coord = uint2(thread % pc.resolution, thread/pc.resolution);

    float2 uv = (float2(coord) + 0.5f) / float2(pc.resolution);
    let values = eval_half_from_inputs(
        inputs_from_textures(pc.latent_textures, uv, float4(0), pc.resolution),
        (half*)pc.params
    );

    pc.output[thread * 3] = float(values[pc.channel_offset]);
    pc.output[thread * 3 + 1] = float(values[pc.channel_offset + 1]);
    pc.output[thread * 3 + 2] = float(values[pc.channel_offset + 2]);
}

[shader("compute")]
[NumThreads(64, 1, 1)]
void render_to_texture(
    uint thread : SV_DispatchThreadID,
    uniform RenderToTexturePushConstants pc
) {
    let coord = uint2(thread % pc.extent.x, thread/pc.extent.x);

    float2 uv_grad = 1 / float2(pc.extent) * pc.scale;
    float2 uv = (float2(coord) + 0.5f) * uv_grad;
    let values = eval_half_from_inputs(
        inputs_from_textures(pc.latent_textures, uv, float4(uv_grad.x, 0, 0, uv_grad.y), pc.texture_size),
        (half*)pc.params
    );

    if (any(thread >= pc.extent.x * pc.extent.y)) {
        return;
    }

    float3 value;
    value[0] = float(values[pc.channel_offset + 0]);
    value[1] = float(values[pc.channel_offset + 1]);
    value[2] = float(values[pc.channel_offset + 2]);

    WTexture2D<float4> output = storage_images[pc.image];
    output.Store(coord, float4(srgb_lin2encoded(value * pow(2, pc.exposure)), 1));
}
