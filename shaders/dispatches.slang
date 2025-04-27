// todo: could be rolled into uniforms but slang-struct-rs isn't good enough yet.
struct DispatchBuffer {
    uint4[4] dispatches;
    Atomic<uint64_t>[2] counts;

    uint num_opaque_meshlets() {
        return dispatches[0].x;
    }

    uint num_alpha_clipped_meshlets() {
        return dispatches[1].x;
    }
}
