import models;
import uniforms;
import prefix_sum;
import dispatches;

uint div_ceil(uint value, uint divisor) {
    return (value + divisor - 1) / divisor;
}

[shader("compute")]
[NumThreads(1, 1, 1)]
void reset_buffers(uniform DispatchBuffer* buffer) {
    buffer.dispatches[0] = {0,1,1};
    buffer.dispatches[1] = {0,1,1};
    buffer.dispatches[2] = {0,1,1};
    buffer.dispatches[3] = {0,1,1};
    buffer.counts[0] = 0;
    buffer.counts[1] = 0;
}

struct PC {
    UniformBuffer* uniforms;
    PrefixSumValue* opaque_values;
    PrefixSumValue* alpha_clipped_values;
}

[shader("compute")]
[NumThreads(64, 1, 1)]
void generate_meshlet_prefix_sums(uniform PC constants, uint thread: SV_DispatchThreadID) {
    let uniforms = constants.uniforms;
    let dispatches = uniforms.dispatches;

    if (thread >= uniforms.num_instances) {
        return;
    }

    let instance = uniforms.instances[thread];
    let model = uniforms.models[instance.model_index];
    let material = model.material;

    uint dispatch_index = uint(material.flags & MaterialFlags::AlphaClipped);

    uint offset;
    InterlockedAdd(dispatches.dispatches[dispatch_index].x, model.num_meshlets, offset);

    uint num_meshlets_so_far = offset + model.num_meshlets;
    InterlockedMax(dispatches.dispatches[dispatch_index + 2].x, div_ceil(num_meshlets_so_far, 64));

    let ptr = dispatch_index > 0 ? constants.alpha_clipped_values : constants.opaque_values;

    PrefixSumBuffer prefix_sum_buffer = {&dispatches.counts[dispatch_index], ptr};

    prefix_sum_inclusive_append(prefix_sum_buffer, thread, model.num_meshlets);
}

struct PC2 {
    UniformBuffer* uniforms;
    PrefixSumValue* values;
    uint is_alpha_clipped_meshlets;
}

[shader("compute")]
[NumThreads(64, 1, 1)]
void write_meshlet_instances(uniform PC2 constants, uint thread: SV_DispatchThreadID) {
    let uniforms = constants.uniforms;
    let dispatches = uniforms.dispatches;

    PrefixSumBuffer prefix_sum_buffer = {&dispatches.counts[constants.is_alpha_clipped_meshlets], constants.values};

    if (thread >= prefix_sum_total(prefix_sum_buffer)) {
        return;
    }

    let val = prefix_sum_binary_search(prefix_sum_buffer, thread);

    let instance = uniforms.instances[val.index];
    let model = uniforms.models[instance.model_index];

    uint32_t local_meshlet_index =
        thread - (val.sum - model.num_meshlets);

    uint output_index = thread;

    if (constants.is_alpha_clipped_meshlets > 0) {
        output_index += dispatches.num_opaque_meshlets();
    }

    uniforms.meshlet_instances[output_index] = {model.meshlets[local_meshlet_index], val.index};
}
