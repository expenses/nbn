import slang.neural;
import ntc.structs;

typealias Address = PointerAddress<float>;
typealias InputVec = InlineVector<float, 16>;
typealias HiddenVec = InlineVector<float, 64>;
typealias OutputVec = InlineVector<float, 16>;

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

[Differentiable]
half[16] eval_network(Network network, no_diff float2 uv, no_diff float mip_level, Address address) {
    let s_1 = network.latent_texture_1.sample(uv, mip_level);
    let s_2 = network.latent_texture_2.sample(uv+0.5, mip_level);
    let s_3 = network.latent_texture_3.sample(uv, mip_level-1);
    let s_4 = network.latent_texture_4.sample(uv+0.5, mip_level-1);
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
    return eval_from_inputs(inputs, address);
}

[Differentiable]
half relu(half x) {
    return max(x, half(0.0));
}

half[16] render(
    int2 pixel,
    int resolution,
    int mip,
    Network network
) {
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    return eval_network(network, uv, mip, Address(network.weights_and_biases));
}

[Differentiable]
float L2(float value, float reference) {
    let diff = value - reference;
    return diff * diff;
}

[Differentiable]
float L1(float value, float reference) {
    let diff = value - reference;
    return abs(diff);
}

float[16] sample_references(
    float2 uv,
    float mip_level,
    Texture2D<float4>[] reference,
    SamplerState samp,
    int num_images
) {
    float[16] values;
    for (int i = 0; i < 16; i++) {
        values[i] = 0.0;
    }
    for (int i = 0; i < num_images; i++) {
        let sample = reference[i].SampleLevel(samp, uv, mip_level).rgb;
        values[i * 3] = sample.r;
        values[i * 3 + 1] = sample.g;
        values[i * 3 + 2] = sample.b;
    }
    return values;
}

[Differentiable]
float[16] loss(
    no_diff float2 uv,
    no_diff float mip_level,
    Texture2D<float4>[] reference,
    SamplerState samp,
    Network network,
    Address address,
    int num_images
) {
    let color = eval_network(network, uv, mip_level, address);
    let reference_values = (no_diff sample_references(uv, mip_level, reference, samp, num_images));

    // todo: system where input images can have variable number of used channels
    let num_channels = num_images * 3;

    float[16] error;
    [ForceUnroll]
    for (int i = 0; i < 16; ++i) {
        error[i] = L2(color[i], reference_values[i]);
        // todo: useful?
        // if (i >= num_channels) {
        //     error[i] = 0.0;
        // }
    }
    return error;
}

struct LCG {
    uint state;

    __init(uint seed) { state = seed; }

    [mutating]
    uint next_uint() {
        const uint A = 1664525u;
        const uint C = 1013904223u;
        state = (A * state + C);
        return state;
    }

    [mutating]
    float next_float() {
        // Convert to float in range [0, 1)
        return (next_uint() >> 8) * 0x1p-24;
    }
};

void optimizer_step(
    inout float primal,
    inout float grad,
    inout float mean,
    inout float variance,
    float learning_rate,
    int iteration
) {
    // Default Adam values.
    const float ADAM_BETA_1 = 0.9;
    const float ADAM_BETA_2 = 0.999;
    const float ADAM_EPSILON = 1e-8;

    // Running average of gradient and square gradient (== variance)
    mean = lerp(grad, mean, ADAM_BETA_1);
    variance = lerp(grad * grad, variance, ADAM_BETA_2);

    // The running average is initialized with zeros, and underestimates
    // the true values at the beginning. Adam divides by a compensation factor
    float mHat = mean / (1.0f - pow(ADAM_BETA_1, iteration));
    float vHat = variance / (1.0f - pow(ADAM_BETA_2, iteration));
    // Compute the adam step by scaling the mean gradient by the learning rate and the variance estimate
    float step = learning_rate * mHat / (sqrt(vHat) + ADAM_EPSILON);

    // Take an adam step
    primal -= step;

    // Because we now accumulate gradients, we need to clear the gradient to zero before the next iteration
    grad = 0.0f;
}


void sum_loss(
    int2 pixel,
    int resolution,
    Texture2D<float4>[] reference,
    SamplerState samp,
    Network network,
    inout float total,
    int num_images
) {
    let loss = loss(
        (float2(pixel) + 0.5f) / float2(resolution),
        0.0f,
        reference,
        samp,
        network,
        Address(network.weights_and_biases),
        num_images
    );
    float local_sum = 0.0;
    for (int i = 0; i < 16; ++i) {
        local_sum += loss[i];
    }
    let wave_sum = WaveActiveSum(local_sum);
    if (WaveIsFirstLane()) {
        InterlockedAdd(total, wave_sum);
    }
}

void calculate_grads(
    uint seed,
    int2 batch_index,
    int2 batch_size,
    Texture2D<float4>[] reference,
    SamplerState samp,
    Network network,
    int num_images
) {
    // Pick a random pixel for computing the loss
    // First, sample a random UV within our cell of the training batch grid
    LCG lcg = LCG(seed);
    float2 uv = (batch_index + float2(lcg.next_float(), lcg.next_float())) / float2(batch_size);
    let num_mip_levels = network.latent_texture_1.num_mip_levels;
    float mip;
    if (lcg.next_float() < 0.1f) {
        mip = (lcg.next_float() * (num_mip_levels+1))-1; // uniform
    } else {
        float u = max(lcg.next_float(), 1e-6f);
        mip = clamp(-log(u) / log(4.0f), 0.0f, float(num_mip_levels - 1));
    }

    // Backpropagate through the loss
    float[16] gradient_seed;
    for (int i = 0; i < 16; ++i)
        gradient_seed[i] = 1.0f;
    let address = DifferentialPtrPair<Address>(Address(network.weights_and_biases), Address(network.weights_and_biases_grad));
    bwd_diff(loss)(uv, mip, reference, samp, network, address, num_images, gradient_seed);
}

uint16_t4 compress_latent_texture(LatentTexture texture, int block) {
    let endpoint_a = quantize(sigmoid(texture.endpoints.load(block * 2)), uint16_t3(5,6,5));
    let endpoint_b = quantize(sigmoid(texture.endpoints.load(block * 2 + 1)), uint16_t3(5,6,5));

    let e1 = endpoint_a.b | (endpoint_a.g << 5) | (endpoint_a.r << 11);
    let e2 = endpoint_b.b | (endpoint_b.g << 5) | (endpoint_b.r << 11);
    // bc1 has a seperate mode if e2 >= e1 that we want to avoid
    let max_e = max(e1, e2);
    let min_e = min(e1, e2);

    uint alphas = 0;
    for (int i = 0; i < 16; ++i) {
        var alpha = uint(quantize(sigmoid(texture.alpha.load(block * 16 + i)), 2));

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

float4 render_tensors(
    LatentTexture texture,
    int2 pixel,
    int2 resolution,
) {
    let uv = (float2(pixel) + 0.5f) / float2(resolution);
    return float4(texture.sample(uv, 0), 1);
}

float4 render_texture(
    Texture2D<float4> texture,
    SamplerState samp,
    int2 pixel,
    int2 resolution,
) {
    let uv = (float2(pixel) + 0.5f) / float2(resolution);
    return texture.SampleLevel(samp, uv, 0);
}

[shader("compute")]
[NumThreads(8, 8, 1)]
void render_compute(
    uint2 thread : SV_DispatchThreadID,
    uniform RenderComputePushConstants pc
) {
    if (any(thread >= uint2(pc.resolution))) {
        return;
    }

    let loc = thread.y * pc.resolution + thread.x;

    let values = render(thread, pc.resolution, 0, *pc.network);

    pc.output[loc * 3] = float(values[0]);
    pc.output[loc * 3 + 1] = float(values[1]);
    pc.output[loc * 3 + 2] = float(values[2]);
}
