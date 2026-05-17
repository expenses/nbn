// todo: could be rolled into uniforms but slang-struct-rs isn't good enough
// yet.
struct DispatchBuffer {
    uint4[2] dispatches;
    Atomic<uint64_t>[2] counts;
    Atomic<uint> visible_meshlets;
}
