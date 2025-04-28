import models;
import uniforms;
import prefix_sum;
import dispatches;
import common;

[shader("compute")][NumThreads(1, 1, 1)] void
reset_buffers(uniform DispatchBuffer* buffer) {
    buffer.dispatches[0] = {0, 1, 1, 0};
    buffer.dispatches[1] = {0, 1, 1, 0};
    buffer.counts[0] = 0;
    buffer.counts[1] = 0;
    buffer.visible_meshlets = 0;
}

struct PC {
    UniformBuffer* uniforms;
}

    [shader("compute")][NumThreads(64, 1, 1)] void
    generate_meshlet_prefix_sums(
        uniform PC constants,
        uint thread : SV_DispatchThreadID
    ) {
    let uniforms = constants.uniforms;
    let dispatches = uniforms.dispatches;

    if (thread >= uniforms.num_instances) {
        return;
    }

    let instance = uniforms.instances[thread];
    let model = uniforms.models[instance.model_index];

    let is_alpha_clipped = model.material.flags & MaterialFlags::AlphaClipped;

    PrefixSumBuffer prefix_sum_buffer =
        uniforms.get_prefix_sum_buffer(is_alpha_clipped);

    let value = prefix_sum_inclusive_append(prefix_sum_buffer, thread, model.num_meshlets);

    InterlockedMax(
        dispatches.dispatches[uint(is_alpha_clipped)].x,
        div_ceil(value.sum, 64)
    );
}
