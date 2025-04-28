// todo: could be rolled into uniforms but slang-struct-rs isn't good enough
// yet.
struct DispatchBuffer {
    uint4[4] dispatches;
    uint2 total_meshlets;
    Atomic<uint64_t>[2] counts;
}
